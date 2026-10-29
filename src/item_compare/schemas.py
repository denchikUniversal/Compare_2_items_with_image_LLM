from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

MAX_CANDIDATES = 100

# Картинка прямо в запросе: data:image/png;base64,... — так веб-интерфейс (web/)
# отправляет перетащенные файлы, которых нет ни по какой ссылке.
DataImageUrl = Annotated[str, StringConstraints(pattern=r"^data:image/[\w.+-]+;base64,")]


class Kind(StrEnum):
    """Какие части двух товаров сравниваются между собой."""

    IMAGE_IMAGE = "image_image"
    TEXT_TEXT = "text_text"
    # Картинка одного товара с текстом другого — когда общих частей нет.
    IMAGE_TEXT = "image_text"


class Item(BaseModel):
    """Товар. Все поля необязательные, но нужна хотя бы картинка или текст.

    Лишние поля (price, sku, ...) молча игнорируются, поэтому можно присылать
    объекты товаров из других сервисов как есть.
    """

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    title: str | None = None
    description: str | None = None
    image_url: HttpUrl | DataImageUrl | None = Field(
        default=None, description="Ссылка http(s) на картинку или сама картинка как data:image/...;base64,..."
    )
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_content(self) -> "Item":
        if self.image_url is None and not self.text():
            raise ValueError("item needs image_url or at least one of title/description/attributes")
        return self

    def text(self) -> str:
        """Текст для модели: название, затем атрибуты, затем описание.

        CLIP читает только ~77 токенов, поэтому самое важное идёт первым,
        а длинное описание обрезается.
        """
        parts = [self.title or ""]
        parts += [f"{key}: {value}" for key, value in self.attributes.items()]
        parts.append(self.description or "")
        return ". ".join(p.strip() for p in parts if p and p.strip())


class ComponentScore(BaseModel):
    kind: Kind
    # Сырой косинус между векторами, от -1 до 1. Полезен для отладки и калибровки.
    cosine: float
    # Откалиброванная похожесть этой части, 0..100.
    similarity: float
    # Доля этой части в итоговом проценте.
    weight: float


class CompareResult(BaseModel):
    similarity: float = Field(description="Итоговая похожесть, 0..100")
    is_same: bool = Field(description="similarity >= порога same_threshold")
    components: list[ComponentScore]


class CompareRequest(BaseModel):
    item_a: Item
    item_b: Item


class MatchRequest(BaseModel):
    query: Item
    candidates: list[Item] = Field(min_length=1, max_length=MAX_CANDIDATES)
    top_k: int | None = Field(default=None, ge=1)


class Match(BaseModel):
    # Позиция кандидата в исходном списке candidates.
    index: int
    id: str | None
    result: CompareResult


class MatchResponse(BaseModel):
    # Отсортировано по убыванию похожести.
    matches: list[Match]
