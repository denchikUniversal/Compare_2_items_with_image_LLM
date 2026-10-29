import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from item_compare.config import Settings
from item_compare.encoders import Encoder
from item_compare.images import fetch_images, make_http_client
from item_compare.schemas import Item
from item_compare.similarity import ItemEmbedding

DEFAULT_CACHE_DIR = Path("data/cache/images")


@dataclass
class Pair:
    a: Item
    b: Item
    label: float  # 1 — один и тот же товар, 0 — разные


def load_pairs(path: Path) -> list[Pair]:
    """Читает JSONL: по строке {"item_a": {...}, "item_b": {...}, "label": 0 или 1}."""
    pairs = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append(Pair(Item.model_validate(row["item_a"]), Item.model_validate(row["item_b"]), float(row["label"])))
    return pairs


def embed_pairs(
    encoder: Encoder, pairs: Sequence[Pair], settings: Settings, cache_dir: Path = DEFAULT_CACHE_DIR
) -> list[tuple[ItemEmbedding, ItemEmbedding]]:
    embs = embed_items(encoder, [p.a for p in pairs] + [p.b for p in pairs], settings, cache_dir)
    return list(zip(embs[: len(pairs)], embs[len(pairs) :]))


def embed_items(
    encoder: Encoder, items: Sequence[Item], settings: Settings, cache_dir: Path = DEFAULT_CACHE_DIR, chunk: int = 256
) -> list[ItemEmbedding]:
    urls = list(dict.fromkeys(str(item.image_url) for item in items if item.image_url))
    texts = list(dict.fromkeys(t for item in items if (t := item.text())))

    # Картинки грузим и кодируем порциями, чтобы не держать весь датасет в памяти.
    image_vecs = []
    for start in range(0, len(urls), chunk):
        images = load_images(urls[start : start + chunk], settings, cache_dir)
        image_vecs.append(encoder.encode_images(images))
    by_url = dict(zip(urls, torch.cat(image_vecs))) if image_vecs else {}
    by_text = dict(zip(texts, encoder.encode_texts(texts)))

    return [
        ItemEmbedding(by_url[str(item.image_url)] if item.image_url else None, by_text.get(item.text()))
        for item in items
    ]


def load_images(urls: Sequence[str], settings: Settings, cache_dir: Path) -> list[Image.Image]:
    """Скачивает картинки один раз и кладёт в cache_dir — повторные запуски идут без сети."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {url: cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.jpg" for url in urls}
    missing = [url for url, path in paths.items() if not path.exists()]
    if missing:
        for url, image in asyncio.run(_download(missing, settings)).items():
            image.save(paths[url], quality=95)
    return [Image.open(paths[url]).convert("RGB") for url in urls]


async def _download(urls: list[str], settings: Settings) -> dict[str, Image.Image]:
    async with make_http_client(settings.image_timeout_s) as client:
        return await fetch_images(client, urls, settings.image_max_bytes)
