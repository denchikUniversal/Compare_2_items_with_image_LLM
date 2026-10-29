import hashlib
import math
import re

import pytest

from RAG.chunker import chunk_document, chunk_id, pack_units, split_markdown_sections
from RAG.config import RAGConfig
from RAG.documents import Chunk, Document, RetrievedChunk
from RAG.llm import MockLLM, create_llm
from RAG.pipeline import RAGPipeline
from RAG.prompt_builder import build_prompt
from RAG.vector_store import QdrantStore


class FakeEmbedder:
    """Мешок слов с хэшированием: без нейросети, но похожие тексты всё равно близки."""

    dim = 256

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def hit(text, source="doc.md", score=0.9):
    return RetrievedChunk(Chunk("id", text, source, "Doc", "policy", 0), score)


@pytest.fixture
def pipeline():
    config = RAGConfig(qdrant_path=None, min_score=0.0)
    p = RAGPipeline(config, embedder=FakeEmbedder(), store=QdrantStore(config.collection_name, None))
    yield p
    p.close()


def test_split_markdown_sections():
    text = "# Title\nintro\n\n## A\nalpha\n## B\nbeta"
    assert split_markdown_sections(text) == [("Title", "intro\n"), ("A", "alpha"), ("B", "beta")]


def test_pack_units_respects_size_and_overlaps():
    units = [f"Sentence number {i}." for i in range(10)]  # по 19-20 символов
    chunks = pack_units(units, chunk_size=60, overlap=25)
    assert all(len(c) <= 60 for c in chunks)
    # Последнее предложение каждого чанка повторяется в начале следующего.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.startswith(prev.split(". ")[-1].rstrip("."))
    assert "Sentence number 9." in chunks[-1]


def test_chunk_ids_are_deterministic():
    assert chunk_id("a.md", 0) == chunk_id("a.md", 0) != chunk_id("a.md", 1)


def test_chunks_get_document_and_section_header():
    doc = Document("r.md", "Returns Policy", "# Returns Policy\n## Electronics\nWithin 15 days.", "policy")
    [chunk] = chunk_document(doc, chunk_size=500, overlap=100)
    assert chunk.text == "Returns Policy > Electronics: Within 15 days."


def test_prompt_numbers_passages_and_handles_empty_context():
    prompt = build_prompt("q?", [hit("first", "a.md"), hit("second", "b.md")])
    assert "[1] (source: a.md)\nfirst" in prompt.user and "[2] (source: b.md)\nsecond" in prompt.user
    assert "no relevant passages" in build_prompt("q?", []).user


def test_mock_llm_answers_from_context_or_says_it_does_not_know():
    llm = MockLLM()
    assert "don't know" in llm.generate(build_prompt("q", []), "q", [])
    chunks = [hit("Shipping > Express: Express shipping costs $14.99. Standard is free.")]
    answer = llm.generate(build_prompt("express cost", chunks), "How much is express shipping?", chunks)
    assert "$14.99" in answer and "[1]" in answer


def test_unknown_llm_name_is_rejected():
    with pytest.raises(ValueError):
        create_llm(RAGConfig(llm="gpt"))


def test_ask_before_ingest_gives_clear_error(pipeline):
    with pytest.raises(RuntimeError, match="ingest"):
        pipeline.ask("anything")


def test_full_flow_on_shop_knowledge_base(pipeline):
    stats = pipeline.ingest()
    assert stats.documents > 5
    assert pipeline.store.count() == len(stats.chunks)

    answer = pipeline.ask("How much does express shipping cost?")
    assert "shipping.md" in [s.chunk.source for s in answer.sources]
    assert "$14.99" in answer.text

    products = pipeline.search("noise cancelling headphones", doc_type="product")
    assert products and all(h.chunk.doc_type == "product" for h in products)
    assert products[0].chunk.metadata["category"] == "audio"


def test_reingest_overwrites_instead_of_duplicating(pipeline):
    first = pipeline.ingest()
    pipeline.ingest()
    assert pipeline.store.count() == len(first.chunks)
