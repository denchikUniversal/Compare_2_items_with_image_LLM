# =============================================================================
# Командная строка для RAG. Запуск из корня репозитория:
#
#   uv run python -m RAG.main demo                      # весь флоу по шагам с пояснениями
#   uv run python -m RAG.main ingest                    # проиндексировать базу знаний
#   uv run python -m RAG.main search "return headphones" [--type policy|product]
#   uv run python -m RAG.main ask "How long is the return window for headphones?" [--show-prompt]
#   uv run python -m RAG.main ask "..." --llm claude    # настоящий Claude (нужен ANTHROPIC_API_KEY)
#   uv run python -m RAG.main chat                      # диалог в терминале
# =============================================================================
import argparse
import textwrap

from RAG.config import RAGConfig
from RAG.documents import RetrievedChunk
from RAG.pipeline import Answer, IngestStats, RAGPipeline

DEMO_QUESTIONS = [
    "How many days do I have to return headphones?",
    "How much does express shipping cost and how fast is it?",
    "Do you sell noise cancelling headphones? How much are they?",
    "What is the capital of France?",
]


# Печатает заголовок шага, чтобы вывод demo читался как учебник.
def step(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# Печатает итоги индексации: сколько документов и чанков, пример чанка и кусок вектора.
def print_ingest(stats: IngestStats, config: RAGConfig) -> None:
    print(f"documents loaded: {stats.documents}")
    print(f"chunks created:   {len(stats.chunks)} (chunk_size={config.chunk_size}, overlap={config.chunk_overlap})")
    example = stats.chunks[0]
    print(f"example chunk #{example.chunk_index} from {example.source}:")
    print(textwrap.indent(textwrap.fill(example.text, 74), "    "))
    shown = ", ".join(f"{x:+.3f}" for x in stats.sample_vector)
    print(f"its embedding: {stats.dim} numbers, first 8 = [{shown}, ...]")
    print(f"saved to Qdrant collection '{config.collection_name}' at {config.qdrant_path or ':memory:'}")


# Печатает найденные чанки со score — главная «отладочная» информация в RAG.
def print_hits(hits: list[RetrievedChunk]) -> None:
    if not hits:
        print("  (nothing above min_score — the LLM will be told there is no context)")
    for i, hit in enumerate(hits, 1):
        preview = hit.chunk.text if len(hit.chunk.text) < 110 else hit.chunk.text[:107] + "..."
        print(f"  [{i}] score={hit.score:.3f}  {hit.chunk.source}\n      {preview}")


# Печатает ответ и, по желанию, полный промпт, который ушёл в LLM.
def print_answer(answer: Answer, show_prompt: bool) -> None:
    if show_prompt:
        print(f"\n--- SYSTEM PROMPT ---\n{answer.prompt.system}\n\n--- USER MESSAGE ---\n{answer.prompt.user}\n---")
    print(f"\nANSWER:\n{textwrap.fill(answer.text, 78)}")


# Демонстрация: индексация и несколько вопросов с выводом каждого шага.
def run_demo(pipeline: RAGPipeline) -> None:
    step("OFFLINE: steps 1-4 — load -> chunk -> embed -> store in Qdrant")
    print_ingest(pipeline.ingest(), pipeline.config)
    for n, question in enumerate(DEMO_QUESTIONS):
        step(f"ONLINE: question {n + 1}/{len(DEMO_QUESTIONS)}: {question}")
        print(f"step 5 — retrieve top-{pipeline.config.top_k} chunks (min_score={pipeline.config.min_score}):")
        answer = pipeline.ask(question)
        print_hits(answer.sources)
        print(f"steps 6-7 — build prompt and generate with '{pipeline.llm.name}' LLM")
        print_answer(answer, show_prompt=n == 0)


# Интерактивный режим: вопросы в цикле, пустая строка или Ctrl+C — выход.
def run_chat(pipeline: RAGPipeline, show_prompt: bool) -> None:
    print("Ask about DemoShop (returns, shipping, warranty, payment, sizes, products). Empty line to quit.")
    while (question := input("\nyou> ").strip()):
        answer = pipeline.ask(question)
        print_hits(answer.sources)
        print_answer(answer, show_prompt)


# Разбирает аргументы командной строки и запускает нужную команду.
def main() -> None:
    parser = argparse.ArgumentParser(description="Educational RAG pipeline over a fake shop knowledge base")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run the whole flow step by step")
    sub.add_parser("ingest", help="index the knowledge base into Qdrant")
    for name in ("search", "ask", "chat"):
        p = sub.add_parser(name)
        if name != "chat":
            p.add_argument("question")
            p.add_argument("--type", choices=["policy", "product"], help="filter by document type")
            p.add_argument("--top-k", type=int)
        if name != "search":
            p.add_argument("--llm", choices=["mock", "claude"], default="mock")
            p.add_argument("--show-prompt", action="store_true", help="print the full prompt sent to the LLM")
    args = parser.parse_args()

    config = RAGConfig(llm=getattr(args, "llm", "mock"))
    try:
        pipeline = RAGPipeline(config)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
    try:
        if args.command == "demo":
            run_demo(pipeline)
        elif args.command == "ingest":
            print_ingest(pipeline.ingest(), config)
        elif args.command == "search":
            print_hits(pipeline.search(args.question, args.top_k, args.type))
        elif args.command == "ask":
            answer = pipeline.ask(args.question, args.top_k, args.type)
            print_hits(answer.sources)
            print_answer(answer, args.show_prompt)
        elif args.command == "chat":
            run_chat(pipeline, args.show_prompt)
    except (KeyboardInterrupt, EOFError):
        pass
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
