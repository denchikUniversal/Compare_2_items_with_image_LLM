import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from item_compare.schemas import Kind


@dataclass(frozen=True)
class Logistic:
    """Переводит сырой косинус в проценты: 100 * sigmoid(scale * (cosine - shift)).

    shift — косинус, при котором похожесть ровно 50%;
    scale — насколько резко растёт процент вокруг этой точки.
    """

    scale: float
    shift: float

    def __call__(self, cosine: float) -> float:
        z = max(-50.0, min(50.0, self.scale * (cosine - self.shift)))
        return 100.0 / (1.0 + math.exp(-z))


# Стартовые значения для openai/clip-vit-base-patch32, подобранные вручную.
# У CLIP разные шкалы, поэтому у каждого вида сравнения своя кривая:
#   картинки: та же фотография с обрезкой/отражением ~0.92-0.98, разные ~0.55-0.76;
#   тексты: перефразированное название того же товара ~0.81-0.98, другой товар ~0.44-0.76,
#           но вариант того же товара (128 GB vs 256 GB) тоже ~0.90 — порогом не отделить;
#   картинка vs текст: совпадение всего ~0.3, несовпадение ~0.1.
# Под свои товары пересчитываются командой `python -m item_compare.training.calibrate`.
DEFAULT_CURVES: dict[Kind, Logistic] = {
    Kind.IMAGE_IMAGE: Logistic(scale=30.0, shift=0.85),
    Kind.TEXT_TEXT: Logistic(scale=20.0, shift=0.80),
    Kind.IMAGE_TEXT: Logistic(scale=50.0, shift=0.25),
}


class Calibration:
    def __init__(self, curves: dict[Kind, Logistic] | None = None) -> None:
        self.curves = {**DEFAULT_CURVES, **(curves or {})}

    def __call__(self, kind: Kind, cosine: float) -> float:
        return self.curves[kind](cosine)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {kind.value: asdict(curve) for kind, curve in self.curves.items()}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(path.read_text())
        return cls({Kind(kind): Logistic(**curve) for kind, curve in data.items()})
