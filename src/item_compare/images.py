import asyncio
import base64
import binascii
import io

import httpx
from PIL import Image, UnidentifiedImageError

USER_AGENT = "item-compare/0.1 (+https://github.com/denchikUniversal/Compare_2_items_with_image_LLM)"


class ImageFetchError(Exception):
    """Картинку по ссылке не удалось скачать или прочитать."""


def make_http_client(timeout_s: float, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout_s,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        transport=transport,
    )


def _label(url: str) -> str:
    """Ссылка для сообщения об ошибке: у data: URI только заголовок, без мегабайтов base64."""
    return url.split(",", 1)[0] + ",..." if url.startswith("data:") else url


async def _download(client: httpx.AsyncClient, url: str, max_bytes: int) -> bytes:
    """Скачивает потоком, обрывая загрузку, если картинка больше max_bytes."""
    buf = bytearray()
    try:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise ImageFetchError(f"{url}: image is larger than {max_bytes} bytes")
    except httpx.HTTPError as exc:
        raise ImageFetchError(f"{url}: {exc}") from exc
    return bytes(buf)


def _decode_data_url(url: str, max_bytes: int) -> bytes:
    try:
        data = base64.b64decode(url.partition(",")[2], validate=True)
    except binascii.Error as exc:
        raise ImageFetchError(f"{_label(url)}: invalid base64") from exc
    if len(data) > max_bytes:
        raise ImageFetchError(f"{_label(url)}: image is larger than {max_bytes} bytes")
    return data


async def fetch_image(client: httpx.AsyncClient, url: str, max_bytes: int) -> Image.Image:
    """Скачивает картинку по ссылке или декодирует её из data: URI (не больше max_bytes)."""
    if url.startswith("data:"):
        data = _decode_data_url(url, max_bytes)
    else:
        data = await _download(client, url, max_bytes)

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ImageFetchError(f"{_label(url)}: not a readable image") from exc
    return image.convert("RGB")


async def fetch_images(client: httpx.AsyncClient, urls: list[str], max_bytes: int) -> dict[str, Image.Image]:
    """Скачивает уникальные ссылки параллельно. Возвращает {url: картинка}."""
    unique = list(dict.fromkeys(urls))
    images = await asyncio.gather(*(fetch_image(client, url, max_bytes) for url in unique))
    return dict(zip(unique, images))
