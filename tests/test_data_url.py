import base64
import io

import httpx
from conftest import COLORS, FakeEncoder, image_server, img
from fastapi.testclient import TestClient
from PIL import Image

from item_compare.api import create_app
from item_compare.config import Settings


def data_url(color: str, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), COLORS[color]).save(buf, fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def compare(client, a: dict, b: dict) -> httpx.Response:
    return client.post("/v1/compare", json={"item_a": a, "item_b": b})


def test_data_url_compares_like_the_same_image_by_link(client):
    by_link = compare(client, {"image_url": img("red")}, {"image_url": img("red_other_photo")}).json()
    inline = compare(client, {"image_url": data_url("red")}, {"image_url": data_url("red_other_photo")}).json()
    mixed = compare(client, {"image_url": data_url("red")}, {"image_url": img("red_other_photo")}).json()
    assert inline == by_link == mixed
    assert inline["is_same"]


def test_data_url_with_text_uses_both_parts(client):
    body = compare(
        client,
        {"title": "red sneaker", "image_url": data_url("red", "JPEG")},
        {"title": "blue phone", "image_url": data_url("blue")},
    ).json()
    assert [c["kind"] for c in body["components"]] == ["image_image", "text_text"]
    assert not body["is_same"]


def test_non_image_data_url_is_rejected_by_schema(client):
    resp = compare(client, {"image_url": "data:text/plain;base64,aGVsbG8="}, {"title": "x"})
    assert resp.status_code == 422


def test_broken_data_url_error_does_not_echo_payload(client):
    garbage = base64.b64encode(b"definitely not an image" * 100).decode()
    for url in (f"data:image/png;base64,{garbage}", "data:image/png;base64,@@not-base64@@"):
        resp = compare(client, {"image_url": url}, {"image_url": img("red")})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail.startswith("data:image/png;base64,...")
        assert len(detail) < 100


def test_data_url_respects_max_bytes():
    app = create_app(
        Settings(_env_file=None, image_max_bytes=50),
        make_encoder=lambda settings: FakeEncoder(),
        http_transport=httpx.MockTransport(image_server),
    )
    with TestClient(app) as client:
        resp = compare(client, {"image_url": data_url("red")}, {"title": "x"})
    assert resp.status_code == 422
    assert "larger than 50 bytes" in resp.json()["detail"]
