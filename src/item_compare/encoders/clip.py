import threading
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from item_compare.config import Settings
from item_compare.encoders.base import resolve_device
from item_compare.heads import ProjectionHeads


class ClipEncoder:
    """CLIP: картинки и английский текст в одном пространстве векторов."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        head_path: Path | None = None,
    ) -> None:
        self.name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim
        self.heads = ProjectionHeads.load(head_path).to(self.device).eval() if head_path else None
        # Модель вызывается из пула потоков FastAPI; один вызов за раз безопаснее для MPS/CUDA.
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "ClipEncoder":
        return cls(settings.model_name, settings.device, settings.batch_size, settings.head_path)

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return self._encode_batched(list(images), self._image_batch)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        return self._encode_batched(list(texts), self._text_batch)

    def _encode_batched(self, inputs: list, encode_batch) -> torch.Tensor:
        if not inputs:
            return torch.empty(0, self.dim)
        with self._lock, torch.inference_mode():
            chunks = [encode_batch(inputs[i : i + self.batch_size]) for i in range(0, len(inputs), self.batch_size)]
        return torch.cat(chunks)

    def _image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        # В transformers 5 get_*_features возвращает объект; вектор лежит в pooler_output.
        emb = F.normalize(self.model.get_image_features(**inputs).pooler_output, dim=-1)
        if self.heads is not None:
            emb = F.normalize(self.heads.image(emb), dim=-1)
        return emb.float().cpu()

    def _text_batch(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        emb = F.normalize(self.model.get_text_features(**inputs).pooler_output, dim=-1)
        if self.heads is not None:
            emb = F.normalize(self.heads.text(emb), dim=-1)
        return emb.float().cpu()
