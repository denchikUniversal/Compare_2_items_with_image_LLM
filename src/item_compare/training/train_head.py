"""Дообучает адаптер (ProjectionHeads) поверх замороженного CLIP на размеченных парах.

    uv run python -m item_compare.training.train_head \\
        --pairs data/pairs.jsonl --out checkpoints/heads.pt

Сам CLIP не обучается: векторы считаются один раз, дальше учится только
маленький адаптер — поэтому обучение быстрое и влезает в 8 ГБ памяти M1.
Адаптер меняет шкалу косинусов, поэтому после обучения обязательно:
    ... -m item_compare.training.calibrate --pairs data/pairs.jsonl --head checkpoints/heads.pt
и запуск сервиса с ITEM_COMPARE_HEAD_PATH и ITEM_COMPARE_CALIBRATION_PATH.

Учатся только сравнения картинка↔картинка и текст↔текст; сравнение картинки с текстом
после обучения стоит перепроверить (calibrate покажет его ROC-AUC).
Следующий шаг, когда данных станет много, — LoRA на сам CLIP (библиотека peft) на GPU.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from item_compare.config import Settings
from item_compare.encoders import ClipEncoder
from item_compare.heads import ProjectionHeads, ResidualHead
from item_compare.similarity import ItemEmbedding
from item_compare.training.data import Pair, embed_pairs, load_pairs
from item_compare.training.losses import contrastive_loss

PairTensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # векторы A, векторы B, метки


def stack(embedded: list[tuple[ItemEmbedding, ItemEmbedding]], pairs: list[Pair], part: str) -> PairTensors | None:
    """Собирает в тензоры пары, у которых есть нужная часть (image или text) с обеих сторон."""
    rows = [
        (getattr(a, part), getattr(b, part), pair.label)
        for (a, b), pair in zip(embedded, pairs)
        if getattr(a, part) is not None and getattr(b, part) is not None
    ]
    if not rows:
        return None
    xa, xb, y = zip(*rows)
    return torch.stack(xa), torch.stack(xb), torch.tensor(y)


def cosines(head: ResidualHead, data: PairTensors) -> torch.Tensor:
    xa, xb, _ = data
    return (F.normalize(head(xa), dim=-1) * F.normalize(head(xb), dim=-1)).sum(-1)


def auc(head: ResidualHead, data: PairTensors | None) -> str:
    if data is None or len(set(data[2].tolist())) < 2:
        return "n/a"
    with torch.no_grad():
        return f"{roc_auc_score(data[2].numpy(), cosines(head, data).numpy()):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL с размеченными парами")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/heads.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--margin", type=float, default=0.5, help="косинус, ниже которого разные пары не штрафуются")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    settings = Settings()
    # Без адаптера: учим его на «чистых» векторах CLIP.
    encoder = ClipEncoder(settings.model_name, settings.device, settings.batch_size)
    pairs = load_pairs(args.pairs)
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_fraction)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    print(f"embedding {len(pairs)} pairs with {settings.model_name} ...")
    train_emb = embed_pairs(encoder, train_pairs, settings)
    val_emb = embed_pairs(encoder, val_pairs, settings)
    heads = ProjectionHeads(encoder.dim, args.hidden)
    parts = {
        part: (stack(train_emb, train_pairs, part), stack(val_emb, val_pairs, part), getattr(heads, part))
        for part in ("image", "text")
    }
    for part, (_, val, head) in parts.items():
        print(f"{part}: val ROC-AUC before training {auc(head, val)}")

    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        heads.train()
        total, steps = 0.0, 0
        for part, (train, _, head) in parts.items():
            if train is None:
                continue
            order = torch.randperm(len(train[2]))
            for start in range(0, len(order), args.batch_size):
                idx = order[start : start + args.batch_size]
                batch = (train[0][idx], train[1][idx], train[2][idx])
                loss = contrastive_loss(cosines(head, batch), batch[2], args.margin)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total, steps = total + loss.item(), steps + 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            heads.eval()
            report = ", ".join(f"{part} val ROC-AUC {auc(head, val)}" for part, (_, val, head) in parts.items())
            print(f"epoch {epoch}: loss {total / max(steps, 1):.4f}; {report}")

    heads.save(args.out)
    print(f"saved {args.out}\nnext: calibrate with --head {args.out}")


if __name__ == "__main__":
    main()
