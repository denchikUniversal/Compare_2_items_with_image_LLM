# =============================================================================
# Структуры данных, которые «текут» по пайплайну:
#   Document (целый источник) -> Chunk (кусок для поиска) -> RetrievedChunk (найденный кусок + score)
# =============================================================================
from dataclasses import dataclass, field
from typing import Any


# Исходный документ до разбиения: целый markdown-файл или один товар из каталога.
@dataclass
class Document:
    # Откуда документ, например "returns_policy.md" или "product:sku-1001".
    source: str
    title: str
    text: str
    # Тип документа — по нему можно фильтровать поиск: "policy" или "product".
    doc_type: str
    # Любые дополнительные поля (цена, бренд...). Попадут в payload Qdrant.
    metadata: dict[str, Any] = field(default_factory=dict)


# Кусок документа (chunk). Именно чанки превращаются в векторы и хранятся в базе.
@dataclass
class Chunk:
    # Уникальный id точки в векторной базе.
    id: str
    # Текст, который видит модель эмбеддингов и который потом уйдёт в промпт LLM.
    text: str
    source: str
    title: str
    doc_type: str
    # Порядковый номер чанка внутри документа.
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


# Чанк, найденный поиском, вместе с оценкой похожести на вопрос (score, чем больше — тем ближе).
@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
