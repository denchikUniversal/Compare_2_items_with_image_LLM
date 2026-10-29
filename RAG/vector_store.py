# =============================================================================
# Шаг 4. ВЕКТОРНАЯ БАЗА ДАННЫХ (Vector Store) — Qdrant
#
# Обычная БД ищет точные совпадения (WHERE category = 'audio'). Векторная БД ищет
# «ближайших соседей» — точки, чьи векторы ближе всего к вектору запроса.
# Чтобы не сравнивать запрос с каждым из миллионов векторов, Qdrant строит индекс
# HNSW — граф «соседей», по которому приближённый поиск идёт за миллисекунды.
#
# Термины Qdrant:
#   collection — набор точек одного вида (как таблица); у неё фиксированы длина
#                вектора и метрика близости (у нас косинус);
#   point      — одна запись: id + vector + payload;
#   payload    — произвольный JSON рядом с вектором (текст чанка, источник, цена...).
#                По payload можно фильтровать поиск: «только товары», «цена < 100».
#
# Как можно запустить клиент (API у всех вариантов одинаковый):
#   QdrantClient(path="RAG/qdrant_data")       — локально в файлах, без сервера (наш вариант);
#   QdrantClient(":memory:")                   — в памяти, данные пропадут (тесты);
#   QdrantClient(url="http://localhost:6333")  — настоящий сервер:
#       docker run -p 6333:6333 qdrant/qdrant  (+ веб-панель на http://localhost:6333/dashboard)
# Локальный режим держит блокировку папки: открыть её может только один процесс за раз.
# =============================================================================
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from RAG.documents import Chunk, RetrievedChunk


# Обёртка над Qdrant, которая знает только о наших Chunk — остальной код
# не зависит от конкретной базы (её можно заменить на Chroma, pgvector, FAISS...).
class QdrantStore:
    # Подключается к базе: в файлах по пути path или в памяти, если path=None.
    def __init__(self, collection_name: str, path: Path | None) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(path=str(path)) if path else QdrantClient(":memory:")

    # Есть ли уже коллекция (то есть делалась ли индексация).
    def exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    # Создаёт коллекцию заново (старые данные удаляются).
    # size — длина вектора, она должна совпадать с размерностью модели эмбеддингов;
    # distance — метрика близости: COSINE, DOT (скалярное произведение) или EUCLID.
    def recreate_collection(self, dim: int) -> None:
        if self.exists():
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # На настоящем сервере для быстрых фильтров по полю создают payload-индекс:
        # self.client.create_payload_index(self.collection_name, "doc_type", "keyword")
        # В локальном режиме индексы не нужны — фильтр работает и так.

    # Записывает чанки с их векторами. upsert = insert or update:
    # точка с уже существующим id перезаписывается, а не дублируется.
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title,
                    "doc_type": chunk.doc_type,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    # Ищет top_k ближайших к запросу точек.
    # min_score отсекает слабые совпадения: лучше не дать LLM ничего, чем дать мусор.
    # doc_type включает фильтр по payload — поиск только среди "policy" или "product".
    def search(
        self, query_vector: list[float], top_k: int, min_score: float, doc_type: str | None = None
    ) -> list[RetrievedChunk]:
        query_filter = (
            Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type))]) if doc_type else None
        )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=min_score,
            with_payload=True,
        )
        return [RetrievedChunk(chunk=self._to_chunk(point.id, point.payload), score=point.score) for point in response.points]

    # Сколько точек лежит в коллекции.
    def count(self) -> int:
        return self.client.count(self.collection_name).count

    # Закрывает соединение и снимает блокировку с папки базы.
    def close(self) -> None:
        self.client.close()

    # Восстанавливает Chunk из payload найденной точки.
    @staticmethod
    def _to_chunk(point_id, payload: dict) -> Chunk:
        base = {"text", "source", "title", "doc_type", "chunk_index"}
        return Chunk(
            id=str(point_id),
            text=payload["text"],
            source=payload["source"],
            title=payload["title"],
            doc_type=payload["doc_type"],
            chunk_index=payload["chunk_index"],
            metadata={k: v for k, v in payload.items() if k not in base},
        )
