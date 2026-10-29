from conftest import img

from item_compare.schemas import MAX_CANDIDATES


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "fake", "device": "cpu"}


def test_same_product_on_different_photos(client):
    resp = client.post(
        "/v1/compare",
        json={
            "item_a": {"title": "Nike Air Max 90 white", "image_url": img("red")},
            "item_b": {"title": "Air Max 90 white Nike", "image_url": img("red_other_photo"), "price": 120},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_same"]
    assert body["similarity"] > 90
    assert [c["kind"] for c in body["components"]] == ["image_image", "text_text"]


def test_different_products(client):
    body = client.post(
        "/v1/compare",
        json={
            "item_a": {"title": "Nike Air Max 90", "image_url": img("red")},
            "item_b": {"title": "Apple iPhone 15 Pro", "image_url": img("blue")},
        },
    ).json()
    assert not body["is_same"]
    assert body["similarity"] < 10


def test_image_only_vs_text_only_uses_cross_comparison(client):
    body = client.post(
        "/v1/compare",
        json={"item_a": {"image_url": img("red")}, "item_b": {"title": "red sneaker"}},
    ).json()
    assert [c["kind"] for c in body["components"]] == ["image_text"]


def test_empty_item_is_rejected(client):
    resp = client.post("/v1/compare", json={"item_a": {"id": "1"}, "item_b": {"title": "x"}})
    assert resp.status_code == 422


def test_unreadable_image_returns_422_with_url(client):
    for name in ("broken", "missing"):
        resp = client.post(
            "/v1/compare",
            json={"item_a": {"image_url": img(name)}, "item_b": {"image_url": img("red")}},
        )
        assert resp.status_code == 422
        assert img(name) in resp.json()["detail"]


def test_match_sorts_candidates_and_applies_top_k(client):
    resp = client.post(
        "/v1/match",
        json={
            "query": {"id": "q", "title": "Nike Air Max 90", "image_url": img("red")},
            "candidates": [
                {"id": "phone", "title": "Apple iPhone 15 Pro", "image_url": img("blue")},
                {"id": "dup", "title": "Nike Air Max 90", "image_url": img("red_other_photo")},
                {"id": "plant", "title": "Monstera plant", "image_url": img("green")},
            ],
            "top_k": 2,
        },
    )
    matches = resp.json()["matches"]
    assert [m["id"] for m in matches] == ["dup", matches[1]["id"]]
    assert [m["index"] for m in matches][0] == 1
    assert len(matches) == 2
    assert matches[0]["result"]["similarity"] >= matches[1]["result"]["similarity"]


def test_match_limits_number_of_candidates(client):
    candidates = [{"title": f"item {i}"} for i in range(MAX_CANDIDATES + 1)]
    resp = client.post("/v1/match", json={"query": {"title": "item"}, "candidates": candidates})
    assert resp.status_code == 422
