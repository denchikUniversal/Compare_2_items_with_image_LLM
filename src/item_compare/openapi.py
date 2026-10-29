"""Схема API в JSON: `uv run python -m item_compare.openapi web/openapi.json`.

Из неё веб-интерфейс генерирует TypeScript-типы (`npm run api:types` в web/),
так что модели Pydantic остаются единственным источником правды о запросах и ответах.
Модель при этом не загружается.
"""

import json
import sys
from pathlib import Path

from item_compare.api import create_app


def main() -> None:
    schema = json.dumps(create_app().openapi(), ensure_ascii=False, indent=2) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(schema)
    else:
        sys.stdout.write(schema)


if __name__ == "__main__":
    main()
