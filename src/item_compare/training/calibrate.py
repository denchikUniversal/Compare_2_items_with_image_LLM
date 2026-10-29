"""Подбирает кривые «косинус → проценты» по размеченным парам.

    uv run python -m item_compare.training.calibrate \\
        --pairs data/pairs.jsonl --out checkpoints/calibration.json [--head checkpoints/heads.pt]

Затем сервис запускается с калибровкой:
    ITEM_COMPARE_CALIBRATION_PATH=checkpoints/calibration.json uv run item-compare-api
"""

import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from item_compare.calibration import Calibration, Logistic
from item_compare.config import Settings
from item_compare.encoders import ClipEncoder
from item_compare.schemas import Kind
from item_compare.similarity import ItemEmbedding, fuse, raw_scores
from item_compare.training.data import embed_pairs, load_pairs

# Меньше примеров каждого класса — кривая остаётся стандартной.
MIN_PER_CLASS = 3


def all_scores(a: ItemEmbedding, b: ItemEmbedding) -> list[tuple[Kind, float]]:
    """Все возможные косинусы пары, включая перекрёстные — больше данных для калибровки."""
    scores = []
    if a.image is not None and b.image is not None:
        scores.append((Kind.IMAGE_IMAGE, float(a.image @ b.image)))
    if a.text is not None and b.text is not None:
        scores.append((Kind.TEXT_TEXT, float(a.text @ b.text)))
    for image, text in ((a.image, b.text), (b.image, a.text)):
        if image is not None and text is not None:
            scores.append((Kind.IMAGE_TEXT, float(image @ text)))
    return scores


def fit_curve(cosines: list[float], labels: list[float]) -> tuple[Logistic, float] | None:
    """Логистическая регрессия на одном признаке: p = sigmoid(w * cos + c) = sigmoid(w * (cos - (-c / w)))."""
    if min(labels.count(0.0), labels.count(1.0)) < MIN_PER_CLASS:
        return None
    # Огромный C = почти без регуляризации: нам нужен настоящий наклон кривой.
    model = LogisticRegression(C=1e6).fit([[c] for c in cosines], labels)
    w, c = float(model.coef_[0][0]), float(model.intercept_[0])
    if w <= 0:
        return None
    return Logistic(scale=w, shift=-c / w), roc_auc_score(labels, cosines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL с размеченными парами")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/calibration.json"))
    parser.add_argument("--head", type=Path, default=None, help="адаптер из train_head, если он есть")
    args = parser.parse_args()

    settings = Settings()
    encoder = ClipEncoder(settings.model_name, settings.device, settings.batch_size, args.head)
    pairs = load_pairs(args.pairs)
    embedded = embed_pairs(encoder, pairs, settings)

    by_kind: dict[Kind, tuple[list[float], list[float]]] = {kind: ([], []) for kind in Kind}
    for (a, b), pair in zip(embedded, pairs):
        for kind, cosine in all_scores(a, b):
            by_kind[kind][0].append(cosine)
            by_kind[kind][1].append(pair.label)

    curves = {}
    for kind, (cosines, labels) in by_kind.items():
        fitted = fit_curve(cosines, labels)
        if fitted is None:
            print(f"{kind}: {len(labels)} pairs — not enough of both classes, keeping the default curve")
            continue
        curves[kind], auc = fitted
        print(f"{kind}: {len(labels)} pairs, ROC-AUC {auc:.3f}, 50% at cosine {curves[kind].shift:.3f}, scale {curves[kind].scale:.1f}")

    calibration = Calibration(curves)
    calibration.save(args.out)

    # Как итоговый процент сервиса разделяет пары с новой калибровкой.
    weights = {Kind.IMAGE_IMAGE: settings.weight_image, Kind.TEXT_TEXT: settings.weight_text, Kind.IMAGE_TEXT: 1.0}
    results = [fuse(raw_scores(a, b), calibration, weights, settings.same_threshold) for a, b in embedded]
    labels = [p.label for p in pairs]
    accuracy = sum(r.is_same == bool(y) for r, y in zip(results, labels)) / len(pairs)
    print(f"overall: accuracy {accuracy:.3f} at threshold {settings.same_threshold}%", end="")
    if len(set(labels)) == 2:
        print(f", ROC-AUC {roc_auc_score(labels, [r.similarity for r in results]):.3f}", end="")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
