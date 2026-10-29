# =============================================================================
# Шаг 1. ЗАГРУЗКА (Load)
# Достаём «сырые» знания из источников и приводим их к единому виду Document.
# В реальных проектах здесь PDF, Confluence, база данных, сайт; у нас —
# markdown-файлы с правилами магазина и JSON-каталог товаров.
# =============================================================================
import json
from pathlib import Path

from RAG.config import RAGConfig
from RAG.documents import Document


# Читает все *.md из папки базы знаний. Строку "# Заголовок" в начале файла берём как title.
def load_markdown_documents(directory: Path) -> list[Document]:
    documents = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first_line = text.splitlines()[0] if text else ""
        title = first_line.lstrip("# ").strip() if first_line.startswith("# ") else path.stem
        documents.append(Document(source=path.name, title=title, text=text, doc_type="policy"))
    return documents


# Собирает текст товара, который «увидит» модель эмбеддингов.
# Важное (название, бренд, категория, цена) — в начале, описание — в конце.
def product_to_text(product: dict) -> str:
    stock = "in stock" if product.get("in_stock") else "out of stock"
    return (
        f"Product: {product['title']}. Brand: {product['brand']}. Category: {product['category']}. "
        f"Price: ${product['price']:.2f} ({stock}). {product.get('description', '')}"
    )


# Превращает каталог товаров в документы: один товар — один документ.
# Структурированные поля (цена, бренд, категория) кладём в metadata,
# чтобы потом фильтровать по ним прямо в векторной базе.
def load_product_documents(path: Path) -> list[Document]:
    products = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            source=f"product:{p['id']}",
            title=p["title"],
            text=product_to_text(p),
            doc_type="product",
            metadata={key: p[key] for key in ("id", "brand", "category", "price", "in_stock")},
        )
        for p in products
    ]


# Загружает все источники из конфига разом.
def load_all(config: RAGConfig) -> list[Document]:
    return load_markdown_documents(config.knowledge_base_dir) + load_product_documents(config.products_path)
