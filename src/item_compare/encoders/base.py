from collections.abc import Sequence
from typing import Protocol

import torch
from PIL import Image


class Encoder(Protocol):
    """Переводит входные данные в нормализованные векторы одного пространства.

    Это точка расширения под «любые данные»: чтобы сравнивать аудио, видео,
    PDF и т.п., добавляется энкодер, который кладёт их в то же пространство
    (своя модель + обучаемая проекция, см. item_compare.heads), а сравнение
    и калибровка остаются прежними.
    """

    name: str
    device: torch.device
    dim: int

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """[N, dim], каждая строка нормализована (длина 1)."""
        ...

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """[N, dim], каждая строка нормализована (длина 1)."""
        ...


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
