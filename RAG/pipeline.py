# =============================================================================
# RAG-ПАЙПЛАЙН — связывает все шаги вместе.
#
# Две независимые части:
#   ИНДЕКСАЦИЯ (offline, один раз и при изменении данных):
#       документы -> чанки -> векторы -> запись в Qdrant
#   ОТВЕТ НА ВОПРОС (online, на каждый запрос):
#       вопрос -> вектор -> поиск в Qdrant -> промпт с контекстом -> LLM -> ответ + источники
# =============================================================================
from dataclasses import dataclass

from RAG.chunker import chunk_documents
from RAG.config import RAGConfig
from RAG.documents import Chunk, RetrievedChunk
from RAG.embedder import Embedder, SentenceTransformerEmbedder
from RAG.llm import LLM, create_llm
from RAG.loader import load_all
from RAG.prompt_builder import Prompt, build_prompt
from RAG.retriever import Retriever
from RAG.vector_store import QdrantStore


# Итог индексации — для вывода и проверки.
@dataclass
class IngestStats:
    documents: int
    chunks: list[Chunk]
    dim: int
    # Первые числа вектора первого чанка — чтобы своими глазами увидеть, что такое эмбеддинг.
    sample_vector: list[float]


# Итог ответа: текст, на каких чанках он основан и какой промпт ушёл в LLM.
@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk]
    prompt: Prompt


class RAGPipeline:
    # Собирает компоненты. Любой из них можно передать снаружи (например, фейковый
    # эмбеддер и базу в памяти в тестах) — иначе он создаётся по конфигу.
    # Этот приём называется dependency injection: части легко заменять независимо.
    def __init__(
        self,
        config: RAGConfig | None = None,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        # LLM создаём первой: если нет ключа для Claude, падаем сразу, ничего не загрузив.
        self.llm = llm or create_llm(self.config)
        self.embedder = embedder or SentenceTransformerEmbedder(self.config.embedding_model)
        self.store = store or QdrantStore(self.config.collection_name, self.config.qdrant_path)
        self.retriever = Retriever(self.embedder, self.store, self.config.top_k, self.config.min_score)

    # ИНДЕКСАЦИЯ: загрузка -> чанки -> векторы -> запись в векторную базу.
    # Коллекция пересоздаётся, поэтому удалённые из папки документы исчезнут и из базы.
    def ingest(self) -> IngestStats:
        documents = load_all(self.config)
        chunks = chunk_documents(documents, self.config.chunk_size, self.config.chunk_overlap)
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        self.store.recreate_collection(self.embedder.dim)
        self.store.upsert(chunks, vectors)
        return IngestStats(documents=len(documents), chunks=chunks, dim=self.embedder.dim, sample_vector=vectors[0][:8])

    # Только поиск, без генерации — удобно, чтобы смотреть, что находит база и с каким score.
    def search(self, question: str, top_k: int | None = None, doc_type: str | None = None) -> list[RetrievedChunk]:
        self._ensure_indexed()
        return self.retriever.retrieve(question, top_k, doc_type)

    # ОТВЕТ НА ВОПРОС: поиск -> промпт -> генерация.
    def ask(self, question: str, top_k: int | None = None, doc_type: str | None = None) -> Answer:
        chunks = self.search(question, top_k, doc_type)
        prompt = build_prompt(question, chunks)
        text = self.llm.generate(prompt, question, chunks)
        return Answer(question=question, text=text, sources=chunks, prompt=prompt)

    # Закрывает базу (снимает блокировку папки Qdrant).
    def close(self) -> None:
        self.store.close()

    # Понятная ошибка вместо падения Qdrant, если забыли проиндексировать данные.
    def _ensure_indexed(self) -> None:
        if not self.store.exists():
            raise RuntimeError("The vector store is empty — run `python -m RAG.main ingest` first.")
