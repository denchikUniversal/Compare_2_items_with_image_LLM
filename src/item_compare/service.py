import asyncio
from collections.abc import Sequence

import httpx

from item_compare.calibration import Calibration
from item_compare.config import Settings
from item_compare.encoders import Encoder
from item_compare.images import fetch_images
from item_compare.schemas import CompareResult, Item, Kind, Match
from item_compare.similarity import ItemEmbedding, fuse, raw_scores


class Comparator:
    """Скачивает картинки, считает векторы и сравнивает товары."""

    def __init__(self, encoder: Encoder, settings: Settings, http: httpx.AsyncClient) -> None:
        self.encoder = encoder
        self.settings = settings
        self.http = http
        self.calibration = Calibration.load(settings.calibration_path) if settings.calibration_path else Calibration()
        self.weights = {
            Kind.IMAGE_IMAGE: settings.weight_image,
            Kind.TEXT_TEXT: settings.weight_text,
            Kind.IMAGE_TEXT: 1.0,
        }

    async def embed(self, items: Sequence[Item]) -> list[ItemEmbedding]:
        """Векторы для всех товаров разом: одинаковые ссылки и тексты считаются один раз."""
        urls = [str(item.image_url) for item in items if item.image_url]
        texts = list(dict.fromkeys(t for item in items if (t := item.text())))
        images = await fetch_images(self.http, urls, self.settings.image_max_bytes)

        # Модель тяжёлая и синхронная — уводим её из event loop в отдельный поток.
        image_vecs = await asyncio.to_thread(self.encoder.encode_images, list(images.values()))
        text_vecs = await asyncio.to_thread(self.encoder.encode_texts, texts)
        by_url = dict(zip(images, image_vecs))
        by_text = dict(zip(texts, text_vecs))

        return [
            ItemEmbedding(
                image=by_url[str(item.image_url)] if item.image_url else None,
                text=by_text.get(item.text()),
            )
            for item in items
        ]

    async def compare(self, a: Item, b: Item) -> CompareResult:
        emb_a, emb_b = await self.embed([a, b])
        return self._result(emb_a, emb_b)

    async def match(self, query: Item, candidates: Sequence[Item], top_k: int | None = None) -> list[Match]:
        """Сравнивает query с каждым кандидатом, самые похожие — первыми."""
        emb_query, *emb_candidates = await self.embed([query, *candidates])
        matches = [
            Match(index=i, id=item.id, result=self._result(emb_query, emb))
            for i, (item, emb) in enumerate(zip(candidates, emb_candidates))
        ]
        matches.sort(key=lambda m: m.result.similarity, reverse=True)
        return matches[:top_k] if top_k else matches

    def _result(self, a: ItemEmbedding, b: ItemEmbedding) -> CompareResult:
        return fuse(raw_scores(a, b), self.calibration, self.weights, self.settings.same_threshold)
