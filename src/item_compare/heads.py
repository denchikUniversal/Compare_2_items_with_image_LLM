from pathlib import Path

import torch
from torch import nn


class ResidualHead(nn.Module):
    """Маленький адаптер поверх замороженного CLIP: x + MLP(x).

    Последний слой инициализирован нулями, поэтому необученная голова
    ничего не меняет, а обучение только «подкручивает» пространство.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ProjectionHeads(nn.Module):
    """Отдельные адаптеры для векторов картинок и текста."""

    def __init__(self, dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.image = ResidualHead(dim, hidden)
        self.text = ResidualHead(dim, hidden)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dim": self.dim, "hidden": self.hidden, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "ProjectionHeads":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        heads = cls(ckpt["dim"], ckpt["hidden"])
        heads.load_state_dict(ckpt["state_dict"])
        return heads
