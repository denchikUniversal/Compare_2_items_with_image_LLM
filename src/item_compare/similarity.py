from dataclasses import dataclass

import torch

from item_compare.calibration import Calibration
from item_compare.schemas import CompareResult, ComponentScore, Kind


@dataclass
class ItemEmbedding:
    """Векторы одного товара. None — у товара нет этой части."""

    image: torch.Tensor | None = None
    text: torch.Tensor | None = None


def raw_scores(a: ItemEmbedding, b: ItemEmbedding) -> list[tuple[Kind, float]]:
    """Выбирает, какие части сравнивать, и считает их косинусы.

    Векторы уже нормализованы, поэтому косинус — просто скалярное произведение.
    """
    pairs: list[tuple[Kind, torch.Tensor, torch.Tensor]] = []
    if a.image is not None and b.image is not None:
        pairs.append((Kind.IMAGE_IMAGE, a.image, b.image))
    if a.text is not None and b.text is not None:
        pairs.append((Kind.TEXT_TEXT, a.text, b.text))
    if not pairs:
        # Общих частей нет: у одного товара только картинка, у другого только текст.
        image, text = (a.image, b.text) if a.image is not None else (b.image, a.text)
        pairs.append((Kind.IMAGE_TEXT, image, text))
    return [(kind, float(x @ y)) for kind, x, y in pairs]


def fuse(
    scores: list[tuple[Kind, float]],
    calibration: Calibration,
    weights: dict[Kind, float],
    threshold: float,
) -> CompareResult:
    """Калибрует каждую часть и складывает их во взвешенный итоговый процент."""
    total_weight = sum(weights[kind] for kind, _ in scores)
    components = [
        ComponentScore(
            kind=kind,
            cosine=round(cosine, 4),
            similarity=round(calibration(kind, cosine), 2),
            weight=round(weights[kind] / total_weight, 4),
        )
        for kind, cosine in scores
    ]
    similarity = round(sum(c.similarity * c.weight for c in components), 2)
    return CompareResult(similarity=similarity, is_same=similarity >= threshold, components=components)
