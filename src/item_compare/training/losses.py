import torch
import torch.nn.functional as F


def contrastive_loss(cosine: torch.Tensor, label: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Классический contrastive loss для пар.

    Одинаковые товары (label=1) тянем к косинусу 1, разные (label=0) отталкиваем,
    пока их косинус выше margin; дальше margin разные пары уже не штрафуются.
    """
    positive = label * (1 - cosine).pow(2)
    negative = (1 - label) * F.relu(cosine - margin).pow(2)
    return (positive + negative).mean()
