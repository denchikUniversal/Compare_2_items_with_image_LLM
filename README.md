# Compare 2 items with image

**[English](#english) · [Українська](#українська)**

## English

A REST API that compares two products (image + text) and returns their similarity as a percentage:
"is this the same product or not". Under the hood it uses PyTorch and the CLIP model, which maps
images and English text into a shared vector space.

- `POST /v1/compare` — similarity of two products.
- `POST /v1/match` — compare a product with a list of candidates and rank them (duplicate search).
- `item_compare.training` — calibrating the percentages and fine-tuning on your own products.
- [`web/`](web/README.md) — a Next.js web UI: drop in two images or two product JSON files and get
  a percentage with a breakdown.
- [`RAG/`](RAG/README.md) — a separate educational simulation of a RAG flow with the Qdrant vector database.

### Running

Prerequisites:

- [uv](https://docs.astral.sh/uv/) for the Python part. It downloads Python 3.12 by itself.
- [Node.js](https://nodejs.org/) 20.9+ with npm — only for the web UI.

#### Python API

From the repository root:

```bash
uv sync                     # dependencies (torch, transformers, fastapi, ...)
uv run item-compare-api     # http://127.0.0.1:8000, Swagger UI: http://127.0.0.1:8000/docs
```

The first run downloads CLIP (~600 MB) into the Hugging Face cache. On Apple Silicon the model runs
on the GPU via MPS, on NVIDIA via CUDA, otherwise on the CPU. The server is ready when the log shows
`Application startup complete`. To use another port: `ITEM_COMPARE_PORT=9000 uv run item-compare-api`.

Check it from a second terminal:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/compare -H 'content-type: application/json' -d @examples/compare_request.json
curl -s -X POST localhost:8000/v1/match   -H 'content-type: application/json' -d @examples/match_request.json
```

#### Web UI

The site runs on top of the Python API, so run `uv sync` from the section above first. Then:

```bash
cd web
npm install
npm run dev                 # Python API on :8000 + site on http://localhost:3000
```

`npm run dev` first regenerates the TypeScript types from the API schema (`npm run api:types`) and
then starts both servers in one terminal. Log lines are prefixed with `api` and `web`; Ctrl+C stops both.

If the API is already running separately (`uv run item-compare-api`), start only the site:

```bash
cd web
npm run dev:web
```

The browser does not call Python directly but Next.js: `/api/*` requests are proxied to the Python API.
The API address comes from `ITEM_COMPARE_API_URL`, or otherwise from `ITEM_COMPARE_HOST` and
`ITEM_COMPARE_PORT` (default `http://127.0.0.1:8000`). You can set it in `web/.env.local`; see
[`web/.env.example`](web/.env.example) for a template.

After changing the models in [schemas.py](src/item_compare/schemas.py), regenerate the site's types:
`cd web && npm run api:types` (`npm run dev` does this automatically).

Production build of the site (start the Python API separately):

```bash
cd web
npm run build
npm run start               # http://localhost:3000
```

All npm scripts, and how the site's types stay in sync with Python, are described in [web/README.md](web/README.md).

#### RAG

```bash
uv run python -m RAG.main demo
```

The other commands are described in [RAG/README.md](RAG/README.md).

### API

A product is a JSON object. All fields are optional, but it needs at least an image or some text.
Extra fields (`price`, `sku`, ...) are ignored, so products from other services can be sent as is.

```json
{
  "id": "sku-1",
  "title": "Nike Air Max 90",
  "description": "Classic sneaker ...",
  "image_url": "https://cdn.example.com/airmax.jpg",
  "attributes": {"brand": "Nike", "color": "white", "size": 42}
}
```

`image_url` is an http(s) link or the image itself as `data:image/...;base64,...`.
The text for the model is assembled as `title`, then `attributes`, then `description`. CLIP reads
only ~77 tokens, so the important parts must come first.

`/v1/compare` response (the same poster in a different photo, with a paraphrased title):

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

- Shared parts are compared: image with image, text with text. If there are no shared parts
  (one product has only a photo, the other only text), the image is compared with the text (`image_text`).
- `cosine` is the raw closeness of the vectors; `similarity` is the same value after calibration, 0–100.
- The total is a weighted average (by default image 0.6, text 0.4). `is_same = similarity >= 50`.
- If an image cannot be downloaded or read, the response is `422` with the image address in `detail`.

`/v1/match` accepts `{query, candidates: [up to 100 products], top_k?}` and returns `matches`
sorted by descending similarity; `index` is the candidate's position in the original list.

### How the percentage is computed

CLIP uses different scales for different kinds of pairs, so the cosine can't simply be multiplied by 100.

| Pair | Cosine (measured with CLIP ViT-B/32) |
|---|---|
| the same photo after crop/mirror/compression | 0.92–0.98 |
| different photos | 0.55–0.76 |
| paraphrased title of the same product | 0.81–0.98 |
| a different product | 0.44–0.76 |
| a variant of the same product (128 GB vs 256 GB) | ~0.90 |
| image vs its own description / someone else's description | ~0.31 / ~0.10 |

Each kind of pair is converted to a percentage by its own logistic curve:
`100 * sigmoid(scale * (cosine - shift))`. The starting curves in
[calibration.py](src/item_compare/calibration.py) were picked by hand. **For your own products,
refit them on labeled pairs** (see below).

Known weak spot: CLIP is poor at telling apart numbers and variants of the same model (storage size,
shoe size, color in the text). Fine-tuning and/or rules based on attributes fix this.

### Settings

Environment variables with the `ITEM_COMPARE_` prefix, or a `.env` file
(full list in [config.py](src/item_compare/config.py)):

| Variable | Default | What it does |
|---|---|---|
| `ITEM_COMPARE_MODEL_NAME` | `openai/clip-vit-base-patch32` | Hugging Face model |
| `ITEM_COMPARE_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `ITEM_COMPARE_WEIGHT_IMAGE` / `_WEIGHT_TEXT` | `0.6` / `0.4` | weight of image and text |
| `ITEM_COMPARE_SAME_THRESHOLD` | `50` | `is_same` threshold |
| `ITEM_COMPARE_CALIBRATION_PATH` | — | JSON produced by `calibrate` |
| `ITEM_COMPARE_HEAD_PATH` | — | adapter produced by `train_head` |
| `ITEM_COMPARE_IMAGE_TIMEOUT_S` / `_IMAGE_MAX_BYTES` | `10` / `10 MB` | image download |
| `ITEM_COMPARE_HOST` / `_PORT` | `127.0.0.1` / `8000` | server address |

### Training on your own products

The data is JSONL, one line per pair of products. `label` is 1 if it is the same product and
0 if they are different. Example: [examples/pairs.example.jsonl](examples/pairs.example.jsonl).

```json
{"item_a": {"title": "...", "image_url": "..."}, "item_b": {"title": "..."}, "label": 1}
```

The most useful negative pairs are similar but different products (neighboring models, different
storage, different color). Meaningful results need hundreds to thousands of pairs.

**1. Calibration** — fast; the model doesn't change, only the "cosine → %" curves are fitted:

```bash
uv run python -m item_compare.training.calibrate --pairs data/pairs.jsonl --out checkpoints/calibration.json
ITEM_COMPARE_CALIBRATION_PATH=checkpoints/calibration.json uv run item-compare-api
```

**2. Adapter fine-tuning.** CLIP stays frozen and only a small head on top of its vectors is trained
([heads.py](src/item_compare/heads.py)), so training fits into 8 GB on an M1. Recalibrate afterwards:

```bash
uv run python -m item_compare.training.train_head --pairs data/pairs.jsonl --out checkpoints/heads.pt
uv run python -m item_compare.training.calibrate  --pairs data/pairs.jsonl --head checkpoints/heads.pt
ITEM_COMPARE_HEAD_PATH=checkpoints/heads.pt ITEM_COMPARE_CALIBRATION_PATH=checkpoints/calibration.json uv run item-compare-api
```

During training, images are downloaded once into `data/cache/images/`. Next steps once there is
more data: LoRA on CLIP itself (the `peft` library) on a GPU, then a larger model.

**Other data types.** Comparison and calibration work on vectors, not on images directly. To compare
audio, video, PDFs, etc., add an encoder that maps such data into the same space (the
[`Encoder`](src/item_compare/encoders/base.py) interface).

### Tests

```bash
uv run pytest                    # fast tests: fake model, no network
RUN_SLOW=1 uv run pytest -m slow # with the real CLIP model
```

### Project structure

```
src/item_compare/
  api.py            FastAPI: /health, /v1/compare, /v1/match
  service.py        Comparator: download images -> vectors -> comparison
  similarity.py     which parts to compare and how to combine the total
  calibration.py    cosine -> percent
  encoders/         Encoder interface and CLIP
  heads.py          trainable adapter on top of CLIP
  images.py         image loading from a URL / data: URI
  schemas.py        Pydantic request and response models
  config.py         settings
  training/         calibrate.py, train_head.py, data.py, losses.py
  openapi.py        exports the API schema as JSON (web/ generates TypeScript types from it)
examples/           sample requests and labeled pairs
RAG/                educational RAG pipeline with Qdrant (has its own README)
tests/
web/
  src/app/          page and the /api/* -> Python API proxy
  src/components/   UI components
  src/lib/api/      API client and types generated from openapi.json
```

### Security

The service downloads images from any URL it receives. If the API is reachable from outside, restrict
the list of allowed domains (otherwise it can be used to reach internal addresses — SSRF) and limit
the request body size (`data:` images arrive inside the JSON).

---

## Українська

REST API, який порівнює два товари (зображення + текст) і повертає схожість у відсотках:
«це той самий товар чи ні». Під капотом PyTorch і модель CLIP, яка переводить зображення
та англійський текст в один векторний простір.

- `POST /v1/compare` — схожість двох товарів.
- `POST /v1/match` — порівняти товар зі списком кандидатів і відсортувати їх (пошук дублікатів).
- `item_compare.training` — калібрування відсотків і донавчання на власних товарах.
- [`web/`](web/README.md) — вебінтерфейс на Next.js: перетягніть два зображення або два JSON-файли
  з товарами й отримайте відсоток із розбором.
- [`RAG/`](RAG/README.md) — окрема навчальна імітація RAG-флоу з векторною базою даних Qdrant.

### Запуск

Що потрібно встановити:

- [uv](https://docs.astral.sh/uv/) — для Python-частини. Python 3.12 він завантажить сам.
- [Node.js](https://nodejs.org/) 20.9+ з npm — лише для вебінтерфейсу.

#### Python API

З кореня репозиторію:

```bash
uv sync                     # залежності (torch, transformers, fastapi, ...)
uv run item-compare-api     # http://127.0.0.1:8000, Swagger UI: http://127.0.0.1:8000/docs
```

Під час першого запуску CLIP (~600 МБ) завантажується в кеш Hugging Face. На Apple Silicon модель
працює на GPU через MPS, на NVIDIA — через CUDA, інакше на CPU. Сервер готовий, коли в логах
з'явиться рядок `Application startup complete`. Інший порт: `ITEM_COMPARE_PORT=9000 uv run item-compare-api`.

Перевірка з другого термінала:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/compare -H 'content-type: application/json' -d @examples/compare_request.json
curl -s -X POST localhost:8000/v1/match   -H 'content-type: application/json' -d @examples/match_request.json
```

#### Вебінтерфейс

Сайт працює поверх Python API, тому спершу виконайте `uv sync` із розділу вище. Далі:

```bash
cd web
npm install
npm run dev                 # Python API на :8000 + сайт на http://localhost:3000
```

`npm run dev` спочатку заново генерує TypeScript-типи зі схеми API (`npm run api:types`), а потім
запускає обидва сервери в одному терміналі. Рядки логів позначені `api` і `web`, Ctrl+C зупиняє обидва.

Якщо API вже запущено окремо (`uv run item-compare-api`), запускайте лише сайт:

```bash
cd web
npm run dev:web
```

Браузер звертається не до Python напряму, а до Next.js: запити `/api/*` проксуються в Python API.
Адреса API береться з `ITEM_COMPARE_API_URL`, а якщо її не задано — з `ITEM_COMPARE_HOST` і
`ITEM_COMPARE_PORT` (за замовчуванням `http://127.0.0.1:8000`). Задати її можна у `web/.env.local`,
зразок — [`web/.env.example`](web/.env.example).

Після зміни моделей у [schemas.py](src/item_compare/schemas.py) оновіть типи для сайту:
`cd web && npm run api:types` (під час `npm run dev` це відбувається автоматично).

Продакшн-збірка сайту (Python API запускається окремо):

```bash
cd web
npm run build
npm run start               # http://localhost:3000
```

Усі npm-скрипти, а також те, як типи сайту синхронізуються з Python, описано у [web/README.md](web/README.md).

#### RAG

```bash
uv run python -m RAG.main demo
```

Інші команди описано в [RAG/README.md](RAG/README.md).

### API

Товар — це JSON-об'єкт. Усі поля необов'язкові, але потрібне хоча б зображення або текст.
Зайві поля (`price`, `sku`, ...) ігноруються, тож товари з інших сервісів можна надсилати як є.

```json
{
  "id": "sku-1",
  "title": "Nike Air Max 90",
  "description": "Classic sneaker ...",
  "image_url": "https://cdn.example.com/airmax.jpg",
  "attributes": {"brand": "Nike", "color": "white", "size": 42}
}
```

`image_url` — посилання http(s) або саме зображення у вигляді `data:image/...;base64,...`.
Текст для моделі складається так: `title`, потім `attributes`, потім `description`. CLIP читає
лише ~77 токенів, тому найважливіше має йти на початку.

Відповідь `/v1/compare` (той самий постер на іншому фото, назву перефразовано):

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

- Порівнюються спільні частини: зображення із зображенням, текст із текстом. Якщо спільних частин
  немає (в одного товару лише фото, в іншого лише текст), зображення порівнюється з текстом (`image_text`).
- `cosine` — сира близькість векторів. `similarity` — та сама величина після калібрування, 0–100.
- Підсумок — зважене середнє (за замовчуванням зображення 0.6, текст 0.4). `is_same = similarity >= 50`.
- Якщо зображення не вдалося завантажити або прочитати, відповідь `422` з адресою зображення в `detail`.

`/v1/match` приймає `{query, candidates: [до 100 товарів], top_k?}` і повертає `matches`
за спаданням схожості; `index` — позиція кандидата у вихідному списку.

### Як обчислюються відсотки

У CLIP різні шкали для різних пар, тому косинус не можна просто помножити на 100.

| Пара | Косинус (виміряно на CLIP ViT-B/32) |
|---|---|
| те саме фото після обрізання/віддзеркалення/стиснення | 0.92–0.98 |
| різні фотографії | 0.55–0.76 |
| перефразована назва того самого товару | 0.81–0.98 |
| інший товар | 0.44–0.76 |
| варіант того самого товару (128 GB vs 256 GB) | ~0.90 |
| зображення vs його власний опис / чужий опис | ~0.31 / ~0.10 |

Кожен вид пари переводиться у відсотки власною логістичною кривою:
`100 * sigmoid(scale * (cosine - shift))`. Стартові криві в
[calibration.py](src/item_compare/calibration.py) підібрано вручну. **Для власних товарів
їх потрібно перерахувати на розмічених парах** (див. нижче).

Відоме слабке місце: CLIP погано розрізняє цифри та варіанти однієї моделі (обсяг пам'яті,
розмір, колір у тексті). Це виправляється донавчанням та/або правилами за атрибутами.

### Налаштування

Змінні середовища з префіксом `ITEM_COMPARE_` або файл `.env`
(повний список у [config.py](src/item_compare/config.py)):

| Змінна | За замовчуванням | Що робить |
|---|---|---|
| `ITEM_COMPARE_MODEL_NAME` | `openai/clip-vit-base-patch32` | модель Hugging Face |
| `ITEM_COMPARE_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `ITEM_COMPARE_WEIGHT_IMAGE` / `_WEIGHT_TEXT` | `0.6` / `0.4` | внесок зображення й тексту |
| `ITEM_COMPARE_SAME_THRESHOLD` | `50` | поріг `is_same` |
| `ITEM_COMPARE_CALIBRATION_PATH` | — | JSON, створений `calibrate` |
| `ITEM_COMPARE_HEAD_PATH` | — | адаптер, створений `train_head` |
| `ITEM_COMPARE_IMAGE_TIMEOUT_S` / `_IMAGE_MAX_BYTES` | `10` / `10 МБ` | завантаження зображень |
| `ITEM_COMPARE_HOST` / `_PORT` | `127.0.0.1` / `8000` | адреса сервера |

### Навчання на власних товарах

Дані — JSONL, по одному рядку на пару товарів. `label` дорівнює 1, якщо це той самий товар,
і 0, якщо різні. Приклад: [examples/pairs.example.jsonl](examples/pairs.example.jsonl).

```json
{"item_a": {"title": "...", "image_url": "..."}, "item_b": {"title": "..."}, "label": 1}
```

Найкорисніші негативні пари — схожі, але різні товари (сусідні моделі, інший обсяг пам'яті,
інший колір). Для змістовного результату потрібні сотні–тисячі пар.

**1. Калібрування** — швидко, модель не змінюється, підбираються лише криві «косинус → %»:

```bash
uv run python -m item_compare.training.calibrate --pairs data/pairs.jsonl --out checkpoints/calibration.json
ITEM_COMPARE_CALIBRATION_PATH=checkpoints/calibration.json uv run item-compare-api
```

**2. Донавчання адаптера.** CLIP заморожений, навчається лише маленька «голова» поверх його векторів
([heads.py](src/item_compare/heads.py)), тому навчання вміщається у 8 ГБ на M1. Після нього
калібрування потрібно перерахувати:

```bash
uv run python -m item_compare.training.train_head --pairs data/pairs.jsonl --out checkpoints/heads.pt
uv run python -m item_compare.training.calibrate  --pairs data/pairs.jsonl --head checkpoints/heads.pt
ITEM_COMPARE_HEAD_PATH=checkpoints/heads.pt ITEM_COMPARE_CALIBRATION_PATH=checkpoints/calibration.json uv run item-compare-api
```

Під час навчання зображення завантажуються один раз у `data/cache/images/`. Наступні кроки, коли
даних стане більше: LoRA на сам CLIP (бібліотека `peft`) на GPU, потім більша модель.

**Інші типи даних.** Порівняння й калібрування працюють із векторами, а не безпосередньо
із зображеннями. Щоб порівнювати аудіо, відео, PDF тощо, додається енкодер, який переводить такі
дані в той самий простір (інтерфейс [`Encoder`](src/item_compare/encoders/base.py)).

### Тести

```bash
uv run pytest                    # швидкі тести: фейкова модель, без мережі
RUN_SLOW=1 uv run pytest -m slow # зі справжньою моделлю CLIP
```

### Структура проєкту

```
src/item_compare/
  api.py            FastAPI: /health, /v1/compare, /v1/match
  service.py        Comparator: завантаження зображень -> вектори -> порівняння
  similarity.py     які частини порівнювати і як скласти підсумок
  calibration.py    косинус -> відсотки
  encoders/         інтерфейс Encoder і CLIP
  heads.py          адаптер, що навчається, поверх CLIP
  images.py         завантаження зображень за посиланням / data: URI
  schemas.py        Pydantic-моделі запитів і відповідей
  config.py         налаштування
  training/         calibrate.py, train_head.py, data.py, losses.py
  openapi.py        вивантаження схеми API в JSON (з неї web/ генерує TypeScript-типи)
examples/           приклади запитів і розмічених пар
RAG/                навчальний RAG-пайплайн із Qdrant (має власний README)
tests/
web/
  src/app/          сторінка та проксі /api/* -> Python API
  src/components/   компоненти інтерфейсу
  src/lib/api/      клієнт API і типи, згенеровані з openapi.json
```

### Безпека

Сервіс завантажує зображення за будь-якими надісланими посиланнями. Якщо API доступний ззовні,
обмежте список дозволених доменів (інакше через нього можна звертатися до внутрішніх адрес — SSRF)
і встановіть ліміт на розмір тіла запиту (зображення в `data:` надходять просто в JSON).
