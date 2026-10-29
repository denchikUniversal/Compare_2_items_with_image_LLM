"""Проверка с настоящей моделью CLIP. Запуск: RUN_SLOW=1 uv run pytest -m slow"""

import os

import pytest
from PIL import Image, ImageOps

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="set RUN_SLOW=1 to load the real CLIP model"),
]


@pytest.fixture(scope="module")
def encoder():
    from item_compare.encoders import ClipEncoder

    return ClipEncoder("openai/clip-vit-base-patch32")


def test_mirrored_photo_is_closer_than_another_picture(encoder):
    gradient = Image.linear_gradient("L").convert("RGB").resize((224, 224))
    checker = Image.new("RGB", (224, 224), "white")
    checker.paste((0, 0, 0), (0, 0, 112, 112))
    checker.paste((0, 0, 0), (112, 112, 224, 224))
    a, mirrored, other = encoder.encode_images([gradient, ImageOps.mirror(gradient), checker])
    assert a.shape == (encoder.dim,)
    assert float(a @ a) == pytest.approx(1.0, abs=1e-4)
    assert float(a @ mirrored) > float(a @ other)


def test_paraphrased_title_is_closer_than_other_product(encoder):
    a, same, other = encoder.encode_texts(
        [
            "Apple iPhone 15 Pro 128GB Natural Titanium",
            "iPhone 15 Pro 128 GB, natural titanium, Apple",
            "Samsung Galaxy S24 Ultra 256GB",
        ]
    )
    assert float(a @ same) > 0.95
    assert float(a @ other) < float(a @ same) - 0.1
