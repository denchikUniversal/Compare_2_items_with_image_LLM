# =============================================================================
# Шаг 2. РАЗБИЕНИЕ НА ЧАНКИ (Chunking)
# Зачем резать документы на куски:
#   1) у модели эмбеддингов лимит длины: MiniLM читает ~256 токенов, остальное молча отбрасывает;
#   2) маленький кусок «про одно» находится поиском точнее, чем целый документ «обо всём»;
#   3) в промпт LLM отдаём только нужные куски, а не всю базу знаний.
# Перекрытие (overlap): конец одного чанка повторяется в начале следующего,
# чтобы мысль, разрезанная на границе, не потерялась.
# =============================================================================
import re
import uuid

from RAG.documents import Chunk, Document


# Разбивает markdown на секции по заголовкам (#, ##, ...).
# Возвращает список пар (заголовок секции, текст секции).
def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading, lines = "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if any(l.strip() for l in lines):
                sections.append((heading, "\n".join(lines)))
            heading, lines = line.lstrip("# ").strip(), []
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((heading, "\n".join(lines)))
    return sections


# Делит текст на минимальные «кирпичики»: каждая строка (пункт списка, строка таблицы)
# и каждое предложение внутри строки. Чанк собирается из целых кирпичиков,
# поэтому предложения никогда не режутся посередине.
def split_units(text: str) -> list[str]:
    units = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            units.extend(s for s in re.split(r"(?<=[.!?])\s+", line) if s)
    return units


# Склеивает кирпичики в чанки длиной не больше chunk_size символов.
# Когда чанк заполнен, следующий начинается с его «хвоста» длиной до overlap символов.
def pack_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len(" ".join([*current, unit])) > chunk_size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            for previous in reversed(current):
                if len(" ".join([previous, *tail])) > overlap:
                    break
                tail.insert(0, previous)
            current = tail
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


# Детерминированный id чанка: одинаковый источник и номер дают одинаковый id.
# Поэтому повторная индексация перезаписывает точки в базе, а не дублирует их.
def chunk_id(source: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}"))


# Режет один документ на чанки. К каждому чанку приписываем «шапку» "Документ > Раздел":
# оторванный от документа кусок «Within 15 days» без шапки непонятно к чему относится,
# а с шапкой "Returns Policy > Electronics" — понятно и модели эмбеддингов, и LLM.
def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    if doc.doc_type == "product":
        # Карточка товара короткая — режем только если она не влезает в чанк.
        pieces = [("", text) for text in pack_units(split_units(doc.text), chunk_size, overlap)]
    else:
        pieces = [
            (heading, text)
            for heading, section in split_markdown_sections(doc.text)
            for text in pack_units(split_units(section), chunk_size, overlap)
        ]

    chunks = []
    for index, (heading, text) in enumerate(pieces):
        header = f"{doc.title} > {heading}" if heading and heading != doc.title else doc.title
        chunks.append(
            Chunk(
                id=chunk_id(doc.source, index),
                text=f"{header}: {text}" if doc.doc_type != "product" else text,
                source=doc.source,
                title=doc.title,
                doc_type=doc.doc_type,
                chunk_index=index,
                metadata=doc.metadata,
            )
        )
    return chunks


# Режет все документы и возвращает общий список чанков.
def chunk_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    return [chunk for doc in documents for chunk in chunk_document(doc, chunk_size, overlap)]
