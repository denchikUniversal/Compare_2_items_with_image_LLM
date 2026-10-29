# RAG: учебная имитация полного флоу

**RAG (Retrieval-Augmented Generation)** — способ заставить LLM отвечать по *вашим* данным,
не переобучая её. Перед генерацией мы **находим** (Retrieval) нужные куски документов,
**дополняем** (Augmented) ими промпт и просим модель **сгенерировать** (Generation) ответ
только по этим кускам.

Здесь всё на примере вымышленного магазина DemoShop: правила возврата, доставки, гарантии,
оплаты, таблица размеров и каталог товаров. Каждый шаг лежит в отдельном файле, над каждым
методом комментарий о том, что он делает и зачем.

## Схема

```
ИНДЕКСАЦИЯ (offline: один раз и при изменении данных)

  data/*.md, products.json
        │  1. loader.py        загрузка -> Document
        ▼
     Document
        │  2. chunker.py       разбиение на куски с перекрытием
        ▼
      Chunk ×31
        │  3. embedder.py      текст -> вектор из 384 чисел (sentence-transformers)
        ▼
  вектор + payload
        │  4. vector_store.py  запись в Qdrant (коллекция shop_knowledge)
        ▼
   ┌─────────┐
   │ Qdrant  │
   └─────────┘
        ▲
        │  5. retriever.py     вопрос -> вектор -> top-k ближайших чанков
        │
ОТВЕТ НА ВОПРОС (online: на каждый запрос)

  вопрос ──► 5. поиск ──► 6. prompt_builder.py ──► 7. llm.py ──► ответ + ссылки [1] [2]
                          контекст + инструкция    Mock или Claude
```

`pipeline.py` связывает шаги, `main.py` — командная строка.

## Запуск

Из корня репозитория (зависимости ставит `uv sync`):

```bash
uv run python -m RAG.main demo      # весь флоу по шагам: чанки, векторы, найденное со score, промпт, ответ
uv run python -m RAG.main ingest    # только индексация
uv run python -m RAG.main search "something for running" --type product   # только поиск
uv run python -m RAG.main ask "How many days do I have to return headphones?" --show-prompt
uv run python -m RAG.main chat      # диалог в терминале
```

По умолчанию генерацию делает `MockLLM`. Это имитация: она работает без сети и ключей и собирает
ответ из предложений найденных чанков. Чтобы ответы писала настоящая модель Claude:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python -m RAG.main ask "Can I return opened earbuds?" --llm claude
```

## Порядок чтения кода

1. [config.py](config.py) — все параметры в одном месте.
2. [documents.py](documents.py) — что течёт по пайплайну: `Document` → `Chunk` → `RetrievedChunk`.
3. [loader.py](loader.py) → [chunker.py](chunker.py) → [embedder.py](embedder.py) →
   [vector_store.py](vector_store.py) → [retriever.py](retriever.py) →
   [prompt_builder.py](prompt_builder.py) → [llm.py](llm.py).
4. [pipeline.py](pipeline.py) — как всё соединяется.
5. [../tests/test_rag.py](../tests/test_rag.py) — тот же флоу на фейковом эмбеддере и базе в памяти.

## Векторная база Qdrant

- **Collection** — как таблица: у всех точек одна длина вектора (384) и метрика (косинус).
- **Point** — запись: `id` + `vector` + `payload` (JSON: текст чанка, источник, цена, бренд...).
- **Поиск** — «ближайшие соседи» к вектору вопроса через индекс HNSW, плюс фильтры по payload
  (`--type product` = `Filter(doc_type == "product")`).
- **upsert** — вставить или перезаписать по `id`. Id чанка детерминированный, поэтому
  повторная индексация не плодит дубли.

Сейчас Qdrant работает в локальном режиме: данные лежат файлами в `RAG/qdrant_data/`, сервер
не нужен. Переход на настоящий сервер — одна строка в [vector_store.py](vector_store.py):

```bash
docker run -p 6333:6333 qdrant/qdrant     # панель: http://localhost:6333/dashboard
```
```python
QdrantClient(url="http://localhost:6333")
```

## Что покрутить для понимания

- `chunk_size` / `chunk_overlap` в [config.py](config.py): как меняется число чанков и качество поиска.
- `min_score`: при 0.0 даже на «What is the capital of France?» в промпт попадёт мусор;
  при 0.6 начнут теряться хорошие ответы.
- `top_k`: больше контекста — больше шанс найти ответ, но и больше шума и токенов.
- Запустите `search` с одним и тем же вопросом разными словами: векторный поиск ищет по смыслу.
- Уберите «шапку» `Документ > Раздел` в [chunker.py](chunker.py) и сравните score.
- Положите свой `.md` в `data/knowledge_base/` и выполните `ingest`.

## Что добавляют в продакшене

Гибридный поиск (векторы + BM25 по ключевым словам), переранжирование найденного cross-encoder'ом,
переформулировку вопроса с учётом истории диалога, оценку качества (набор вопросов с эталонными
ответами) и обновление индекса по изменениям, а не целиком.
