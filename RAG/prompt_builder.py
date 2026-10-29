# =============================================================================
# Шаг 6. ДОПОЛНЕНИЕ ПРОМПТА (Augmentation) — буква "A" в RAG.
# LLM ничего не знает о нашем магазине. Мы «дополняем» её знания: вставляем
# найденные чанки прямо в промпт и просим отвечать только по ним.
# Это и есть главная идея RAG: модель не переобучают — ей дают нужные факты на входе.
# =============================================================================
from dataclasses import dataclass

from RAG.documents import RetrievedChunk

SYSTEM_PROMPT = """You are a customer support assistant for DemoShop, an online store.
Answer the customer's question using only the numbered context passages you are given.
After each fact, cite the passage it came from, like [1] or [2].
If the passages do not contain the answer, say you don't know and suggest contacting support at support@demoshop.example — never invent policies, prices or product details."""


# Готовый промпт: системная инструкция (роль и правила) + сообщение пользователя (контекст + вопрос).
@dataclass
class Prompt:
    system: str
    user: str


# Превращает найденные чанки в пронумерованный контекст.
# Номер [n] — это то, на что LLM сошлётся в ответе; по номеру мы покажем пользователю источник.
def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] (source: {rc.chunk.source})\n{rc.chunk.text}" for i, rc in enumerate(chunks, 1))


# Собирает итоговый промпт. Если поиск ничего не нашёл, честно говорим об этом модели —
# по системной инструкции она ответит «не знаю» вместо выдумки (защита от галлюцинаций).
def build_prompt(question: str, chunks: list[RetrievedChunk]) -> Prompt:
    context = format_context(chunks) if chunks else "(no relevant passages were found)"
    user = f"Context passages:\n\n{context}\n\nCustomer question: {question}"
    return Prompt(system=SYSTEM_PROMPT, user=user)
