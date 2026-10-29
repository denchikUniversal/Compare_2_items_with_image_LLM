import hashlib
import io
import re
from collections.abc import Sequence

import httpx
import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image

from item_compare.api import create_app
from item_compare.config import Settings

# Картинки, которые «отдаёт» подставной сервер: http://img.test/<name>.png
COLORS = {
    "red": (220, 20, 20),
    "red_other_photo": (200, 30, 25),
    "blue": (20, 20, 220),
    "green": (20, 200, 20),
}


class FakeEncoder:
    """Энкодер без нейросети, чтобы тесты были быстрыми и детерминированными.

    Вектор картинки — её пиксели 4x4 (48 чисел), вектор текста — счётчик хэшей слов.
    """

    name = "fake"
    device = torch.device("cpu")
    dim = 48

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty(0, self.dim)
        pixels = [torch.tensor(list(img.resize((4, 4)).tobytes()), dtype=torch.float32) for img in images]
        return F.normalize(torch.stack(pixels), dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        vecs = torch.zeros(len(texts), self.dim)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vecs[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return F.normalize(vecs, dim=-1)


def image_server(request: httpx.Request) -> httpx.Response:
    name = request.url.path.strip("/").removesuffix(".png")
    if name == "broken":
        return httpx.Response(200, content=b"definitely not an image")
    if name not in COLORS:
        return httpx.Response(404)
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), COLORS[name]).save(buf, "PNG")
    return httpx.Response(200, content=buf.getvalue(), headers={"content-type": "image/png"})


def img(name: str) -> str:
    return f"http://img.test/{name}.png"


@pytest.fixture
def client():
    app = create_app(
        Settings(_env_file=None),
        make_encoder=lambda settings: FakeEncoder(),
        http_transport=httpx.MockTransport(image_server),
    )
    with TestClient(app) as c:
        yield c
