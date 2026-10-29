# =============================================================================
# Шаг 3. ЭМБЕДДИНГИ (Embeddings)
# Эмбеддинг — это вектор (список чисел), в котором близкие по смыслу тексты
# оказываются рядом. «How many days to return headphones?» и «Electronics can be
# returned within 15 days» получат похожие векторы, хотя общих слов почти нет.
# Именно поэтому векторный поиск находит ответ по смыслу, а не по ключевым словам.
# =============================================================================
from typing import Protocol

from sentence_transformers import SentenceTransformer


# Интерфейс любого эмбеддера. Пайплайну всё равно, какая модель внутри:
# можно подменить на OpenAI/Voyage/свою модель или на фейковый эмбеддер в тестах.
class Embedder(Protocol):
    # Длина вектора — нужна, чтобы создать коллекцию в векторной базе.
    dim: int

    # Векторы для чанков (вызывается при индексации).
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    # Вектор для вопроса пользователя (вызывается на каждый вопрос).
    def embed_query(self, text: str) -> list[float]: ...


# Эмбеддер на библиотеке sentence-transformers (работает локально, без API-ключей).
class SentenceTransformerEmbedder:
    # Загружает модель. При первом запуске она скачивается в кэш Hugging Face (~90 МБ).
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    # Векторы для документов. normalize_embeddings=True приводит длину вектора к 1:
    # тогда косинусная похожесть — это просто скалярное произведение векторов.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode_document(texts, normalize_embeddings=True).tolist()

    # Вектор для вопроса. Отдельный метод нужен потому, что некоторые модели (e5, bge)
    # ждут разные префиксы для вопросов и для документов; encode_query учитывает это сам.
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode_query(text, normalize_embeddings=True).tolist()
