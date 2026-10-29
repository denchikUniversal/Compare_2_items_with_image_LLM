# Item Compare — веб-интерфейс

Перетащите две картинки или два JSON-файла с товарами (можно картинку против JSON) и получите
похожесть в процентах. Считает её Python-API из корня репозитория, этот проект на Next.js —
только интерфейс к нему.

- Кольцо с итоговым процентом и вердиктом «один и тот же товар / разные».
- Разбор: из каких частей сложился результат (картинка↔картинка, текст↔текст, картинка↔текст),
  их косинусы, веса и формула итога.
- Для двух JSON — таблица полей: что совпало, что отличается и какие поля Python не учитывает.
- Два файла, брошенные разом, сами разложатся по слотам A и B. Из буфера (`⌘V`) можно вставить
  картинку, JSON-текст или ссылку на картинку.

## Запуск

Нужны Node.js 20+ и [uv](https://docs.astral.sh/uv/) (как для Python-части).

```bash
cd web
npm install
npm run dev        # Python-API на :8000 и Next.js на http://localhost:3000 одной командой
```

Python-API уже запущен отдельно (`uv run item-compare-api`)? Тогда достаточно `npm run dev:web`.
Индикатор в правом верхнем углу показывает, отвечает ли API и какая модель загружена. Пока модель
грузится (первый раз — ещё и скачивается), он красный.

## Как устроена синхронизация с Python

```
браузер ──/api/*──▶ Next.js (src/app/api/[...path]/route.ts) ──▶ Python FastAPI
```

- **Один источник правды — Pydantic-модели** в `src/item_compare/schemas.py`.
  `npm run api:types` выгружает из них OpenAPI-схему (`python -m item_compare.openapi` → `openapi.json`)
  и генерирует TypeScript-типы (`src/lib/api/schema.ts`). Запросы идут через типизированный клиент
  [openapi-fetch](https://openapi-ts.dev/openapi-fetch/), так что неверное поле или путь — ошибка компиляции.
  Если в Python появится новый `Kind`, сборка упадёт в `Breakdown.tsx`, пока для него не добавят подпись.
- **Список полей товара тоже берётся из схемы**: всё, чего нет в `Item`, помечается «Python не учитывает».
- **Картинки уходят в API как `data:image/...;base64,...`** в том же поле `image_url`. Python декодирует
  их с тем же лимитом `ITEM_COMPARE_IMAGE_MAX_BYTES`, что и картинки по ссылке. Поэтому перетащенный
  файл и та же картинка по URL дают одинаковый результат.
- **Валидирует только Python.** JSON уходит как есть, а ошибки Pydantic показываются по-русски с
  указанием слота (A/B) и поля.
- **Адрес API берётся из тех же переменных окружения, что у Python**: `ITEM_COMPARE_HOST` и
  `ITEM_COMPARE_PORT` (по умолчанию `127.0.0.1:8000`). Можно задать адрес целиком через
  `ITEM_COMPARE_API_URL` — см. `.env.example`.

После любого изменения схем в Python:

```bash
npm run api:types   # перегенерировать openapi.json и schema.ts
npm run typecheck   # TypeScript покажет, что во фронтенде разошлось с API
```

`npm run dev` перегенерирует типы сам при каждом старте. В CI пригодится `npm run api:check`: он падает,
если закоммиченные `openapi.json`/`schema.ts` отстали от Python.

## Скрипты

| Команда | Что делает |
| --- | --- |
| `npm run dev` | перегенерирует типы, запускает Python-API и Next.js вместе |
| `npm run dev:web` / `dev:api` | только Next.js / только Python-API |
| `npm run api:types` | `openapi.json` и `src/lib/api/schema.ts` из Pydantic-моделей |
| `npm run api:check` | то же + `git diff --exit-code`: схема в git совпадает с Python |
| `npm run typecheck` | `next typegen` + `tsc --noEmit` |
| `npm run lint` | ESLint |
| `npm run build` / `start` | продакшн-сборка и запуск (Python-API запускается отдельно) |

## Структура

```
src/app/page.tsx                 страница
src/app/api/[...path]/route.ts   прокси в Python-API с понятной ошибкой, если он не запущен
src/components/Comparer.tsx      состояние: слоты A/B, drag & drop, вставка, запрос сравнения
src/components/…                 DropSlot, SlotPreview, ScoreRing, Breakdown, FieldDiff, ApiStatus
src/lib/api/                     сгенерированные типы, клиент и перевод ошибок API
src/lib/files.ts                 файл → data: URI или JSON-объект
src/lib/diff.ts                  сравнение двух JSON поле за полем
openapi.json                     схема API (генерируется, не править руками)
```
