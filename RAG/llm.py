# =============================================================================
# Шаг 7. ГЕНЕРАЦИЯ (Generation) — буква "G" в RAG.
# LLM получает промпт с контекстом и пишет ответ человеческим языком.
# Здесь две реализации с одинаковым интерфейсом:
#   MockLLM   — имитация: работает без сети, ключей и денег, показывает поток данных;
#   ClaudeLLM — настоящая модель Claude через Anthropic API (нужен ANTHROPIC_API_KEY).
# =============================================================================
import re
from typing import Protocol

import anthropic

from RAG.config import RAGConfig
from RAG.documents import RetrievedChunk
from RAG.prompt_builder import Prompt

# Служебные слова, которые не помогают понять, о чём вопрос.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "do", "does", "can", "i", "my", "you", "your", "to", "of", "for",
    "in", "on", "and", "or", "how", "what", "which", "when", "much", "many", "it", "if", "with", "be",
}


# Интерфейс генератора. question и chunks передаются отдельно для MockLLM,
# настоящей модели достаточно prompt — в нём уже всё есть.
class LLM(Protocol):
    name: str

    # Возвращает текст ответа.
    def generate(self, prompt: Prompt, question: str, chunks: list[RetrievedChunk]) -> str: ...


# Имитация LLM. Никакого «понимания» нет: берём из найденных чанков предложения,
# у которых больше всего общих слов с вопросом, и склеиваем их в ответ со ссылками [n].
# Этого достаточно, чтобы увидеть весь RAG-флоу целиком, ничего не настраивая.
class MockLLM:
    name = "mock"

    # Собирает «ответ» из лучших предложений найденных чанков.
    def generate(self, prompt: Prompt, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "[mock] I don't know — the knowledge base has nothing relevant. Please contact support@demoshop.example."
        question_words = self._words(question)
        scored = []
        for number, rc in enumerate(chunks, 1):
            # Отрезаем «шапку» чанка (до первого ': '), отвечаем только содержимым.
            body = rc.chunk.text.split(": ", 1)[-1]
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                overlap = len(question_words & self._words(sentence))
                if overlap:
                    scored.append((overlap, rc.score, sentence.strip(), number))
        if not scored:
            return f"[mock] The closest passage is [1]: {chunks[0].chunk.text}"
        best = sorted(scored, key=lambda s: (s[0], s[1]), reverse=True)[:2]
        return "[mock] " + " ".join(f"{sentence} [{number}]" for _, _, sentence, number in best)

    # Значимые слова текста в нижнем регистре.
    @staticmethod
    def _words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS}


# Настоящая LLM — Claude через официальный SDK anthropic.
# Ключ берётся из переменной окружения ANTHROPIC_API_KEY (в код его не пишем!).
class ClaudeLLM:
    name = "claude"

    # Создаёт клиента. model — какая модель Claude, effort — насколько глубоко ей думать.
    # Клиент сам ищет ключ: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN или профиль `ant auth login`.
    def __init__(self, model: str, effort: str) -> None:
        self.client = anthropic.Anthropic()
        if not (self.client.api_key or self.client.auth_token or self.client.credentials):
            raise RuntimeError("Claude credentials not found: export ANTHROPIC_API_KEY=... (or run `ant auth login`).")
        self.model = model
        self.effort = effort

    # Отправляет промпт в API и возвращает текст ответа.
    # fallbacks="default": если модель откажется отвечать по соображениям безопасности,
    # API само повторит запрос на подходящей запасной модели в рамках того же вызова.
    def generate(self, prompt: Prompt, question: str, chunks: list[RetrievedChunk]) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return "The model declined to answer this question."
        # В ответе бывают разные блоки (например, thinking) — берём только текст.
        return "".join(block.text for block in response.content if block.type == "text")


# Фабрика: по имени из конфига ("mock" или "claude") создаёт нужный генератор.
def create_llm(config: RAGConfig) -> LLM:
    if config.llm == "mock":
        return MockLLM()
    if config.llm == "claude":
        return ClaudeLLM(config.claude_model, config.claude_effort)
    raise ValueError(f"unknown llm {config.llm!r}, expected 'mock' or 'claude'")
