# =============================================================================
# Шаг 5. ПОИСК (Retrieval) — буква "R" в RAG.
# Вопрос пользователя превращаем в вектор ТОЙ ЖЕ моделью, что и документы,
# и просим векторную базу найти ближайшие чанки.
# Здесь же место для улучшений поиска: гибридный поиск (векторы + ключевые слова BM25),
# переранжирование найденного cross-encoder'ом, переформулировка вопроса через LLM.
# =============================================================================
from RAG.documents import RetrievedChunk
from RAG.embedder import Embedder
from RAG.vector_store import QdrantStore


# Связывает эмбеддер и векторную базу в одно действие «вопрос -> релевантные чанки».
class Retriever:
    # top_k и min_score по умолчанию берутся из конфига, но их можно переопределить в retrieve().
    def __init__(self, embedder: Embedder, store: QdrantStore, top_k: int, min_score: float) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    # Вопрос -> вектор -> ближайшие чанки (отсортированы по убыванию score).
    def retrieve(self, question: str, top_k: int | None = None, doc_type: str | None = None) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_query(question)
        return self.store.search(query_vector, top_k or self.top_k, self.min_score, doc_type)
