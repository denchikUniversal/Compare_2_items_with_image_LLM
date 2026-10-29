# Compare 2 items with image

REST API, который сравнивает два товара (картинка + текст) и возвращает похожесть в процентах:
«это один и тот же товар или нет». Под капотом PyTorch и модель CLIP, которая кладёт картинки
и английский текст в одно пространство векторов.

- `POST /v1/compare` — похожесть двух товаров.
- `POST /v1/match` — сравнить товар со списком кандидатов и отсортировать их (поиск дублей).
- `item_compare.training` — калибровка процентов и дообучение под свои товары.
- [`RAG/`](RAG/README.md) — отдельная учебная имитация RAG-флоу с векторной базой Qdrant.
- Web UI: см. [`web/README.md`](web/README.md).

## Быстрый старт

Нужен [uv](https://docs.astral.sh/uv/). Python 3.12 он скачает сам.

```bash
uv sync                     # зависимости (torch, transformers, fastapi, ...)
uv run item-compare-api     # http://127.0.0.1:8000, Swagger UI: http://127.0.0.1:8000/docs
```

Первый запуск скачивает CLIP (~600 МБ) в кэш Hugging Face. На Apple Silicon модель работает
на GPU через MPS, на NVIDIA — через CUDA, иначе на CPU.

```bash
curl -s -X POST localhost:8000/v1/compare -H 'content-type: application/json' -d @examples/compare_request.json
curl -s -X POST localhost:8000/v1/match   -H 'content-type: application/json' -d @examples/match_request.json
```

## API

Товар — JSON-объект. Все поля необязательные, но нужна хотя бы картинка или текст.
Лишние поля (`price`, `sku`, ...) игнорируются, поэтому товары из других сервисов можно присылать как есть.

```json
{
  "id": "sku-1",
  "title": "Nike Air Max 90",
  "description": "Classic sneaker ...",
  "image_url": "https://cdn.example.com/airmax.jpg",
  "attributes": {"brand": "Nike", "color": "white", "size": 42}
}
```

`image_url` — ссылка http(s) или сама картинка как `data:image/...;base64,...`.
Текст для модели собирается так: `title`, затем `attributes`, затем `description`. CLIP читает
только ~77 токенов, поэтому важное должно быть в начале.

Ответ `/v1/compare` (тот же постер на другом фото, название перефразировано):

```json
{
  "similarity": 76.41,
  "is_same": true,
  "components": [
    {"kind": "image_image", "cosine": 0.921, "similarity": 89.39, "weight": 0.6},
    {"kind": "text_text",   "cosine": 0.814, "similarity": 56.95, "weight": 0.4}
  ]
}
```

- Сравниваются общие части: картинка с картинкой, текст с текстом. Если общих частей нет
  (у одного только фото, у другого только текст), сравнивается картинка с текстом (`image_text`).
- `cosine` — сырая близость векторов. `similarity` — она же после калибровки, 0–100.
- Итог — взвешенное среднее (по умолчанию картинка 0.6, текст 0.4). `is_same = similarity >= 50`.
- Если картинку не удалось скачать или прочитать, ответ `422` с адресом картинки в `detail`.

`/v1/match` принимает `{query, candidates: [до 100 товаров], top_k?}` и возвращает
`matches` по убыванию похожести; `index` — позиция кандидата в исходном списке.

## Как получаются проценты

У CLIP разные шкалы для разных пар, поэтому косинус нельзя просто умножить на 100.

| Пара | Косинус (замеры на CLIP ViT-B/32) |
|---|---|
| то же фото после обрезки/отражения/сжатия | 0.92–0.98 |
| разные фотографии | 0.55–0.76 |
| перефразированное название того же товара | 0.81–0.98 |
| другой товар | 0.44–0.76 |
| вариант того же товара (128 GB vs 256 GB) | ~0.90 |
| картинка vs её описание / чужое описание | ~0.31 / ~0.10 |

Каждый вид пары переводится в проценты своей логистической кривой:
`100 * sigmoid(scale * (cosine - shift))`. Стартовые кривые в
[calibration.py](src/item_compare/calibration.py) подобраны вручную. **Для своих товаров
их нужно пересчитать на размеченных парах** (см. ниже).

Известное слабое место: CLIP плохо различает цифры и варианты одной модели (объём памяти,
размер, цвет в тексте). Это лечится дообучением и/или правилами по атрибутам.

## Настройки

Переменные окружения с префиксом `ITEM_COMPARE_` или файл `.env`
(полный список в [config.py](src/item_compare/config.py)):

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `ITEM_COMPARE_MODEL_NAME` | `openai/clip-vit-base-patch32` | модель Hugging Face |
| `ITEM_COMPARE_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `ITEM_COMPARE_WEIGHT_IMAGE` / `_WEIGHT_TEXT` | `0.6` / `0.4` | вклад картинки и текста |
| `ITEM_COMPARE_SAME_THRESHOLD` | `50` | порог `is_same` |
| `ITEM_COMPARE_CALIBRATION_PATH` | — | JSON из `calibrate` |
| `ITEM_COMPARE_HEAD_PATH` | — | адаптер из `train_head` |
| `ITEM_COMPARE_IMAGE_TIMEOUT_S` / `_IMAGE_MAX_BYTES` | `10` / `10 МБ` | загрузка картинок |
| `ITEM_COMPARE_HOST` / `_PORT` | `127.0.0.1` / `8000` | адрес сервера |

## Обучение под свои товары

Данные — JSONL, по строке на пару товаров. `label` равен 1, если это один и тот же товар,
и 0, если разные. Пример: [examples/pairs.example.jsonl](examples/pairs.example.jsonl).

```json
{"item_a": {"title": "...", "image_url": "..."}, "item_b": {"title": "..."}, "label": 1}
```

Самые полезные отрицательные пары — похожие, но разные товары (соседние модели, другой объём
памяти, другой цвет). Для осмысленного результата нужны сотни–тысячи пар.

**1. Калибровка** — быстро, модель не меняется, подбираются только кривые «косинус → %»:

```bash
uv run python -m item_compare.training.calibrate --pairs data/pairs.jsonl --out checkpoints/calibration.json
ITEM_COMPARE_CALIBRATION_PATH=checkpoints/calibration.json uv run item-compare-api
```

**2. Дообучение адаптера.** CLIP заморожен, учится маленькая голова поверх его векторов
([heads.py](src/item_compare/heads.py)), поэтому обучение влезает в 8 ГБ на M1. После него
калибровку нужно пересчитать:

```bash
uv run python -m item_compare.training.train_head --pairs data/pairs.jsonl --out checkpoints/heads.pt
uv run python -m item_compare.training.calibrate  --pairs data/pairs.jsonl --head checkpoints/heads.pt
ITEM_COMPARE_HEAD_PATH=checkpoints/heads.pt ITEM_COMPARE_CALIBRATION_PATH=checkpoints/calibration.json uv run item-compare-api
```

Картинки при обучении скачиваются один раз в `data/cache/images/`. Следующие шаги, когда данных
станет много: LoRA на сам CLIP (библиотека `peft`) на GPU, затем более крупная модель.

**Другие типы данных.** Сравнение и калибровка работают с векторами, а не с картинками напрямую.
Чтобы сравнивать аудио, видео, PDF и т.д., добавляется энкодер, который кладёт такие данные
в то же пространство (интерфейс [`Encoder`](src/item_compare/encoders/base.py)).

## Тесты

```bash
uv run pytest                    # быстрые тесты: фейковая модель, без сети
RUN_SLOW=1 uv run pytest -m slow # с настоящим CLIP
```

## Структура

```
src/item_compare/
  api.py            FastAPI: /health, /v1/compare, /v1/match
  service.py        Comparator: загрузка картинок -> векторы -> сравнение
  similarity.py     какие части сравнивать и как сложить итог
  calibration.py    косинус -> проценты
  encoders/         Encoder-интерфейс и CLIP
  heads.py          обучаемый адаптер поверх CLIP
  images.py         загрузка картинок по ссылке / data: URI
  schemas.py        Pydantic-модели запросов и ответов
  config.py         настройки
  training/         calibrate.py, train_head.py, data.py, losses.py
examples/           примеры запросов и размеченных пар
RAG/                учебный RAG-пайплайн с Qdrant (свой README)
tests/
```

## Безопасность

Сервис скачивает картинки по любым присланным ссылкам. Если API доступен извне, ограничьте
список разрешённых доменов (иначе через него можно обращаться к внутренним адресам — SSRF)
и поставьте лимит на размер тела запроса (картинки в `data:` приходят прямо в JSON).
