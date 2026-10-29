from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from item_compare import __version__
from item_compare.config import Settings, get_settings
from item_compare.encoders import ClipEncoder, Encoder
from item_compare.images import ImageFetchError, make_http_client
from item_compare.schemas import CompareRequest, CompareResult, MatchRequest, MatchResponse
from item_compare.service import Comparator


def get_comparator(request: Request) -> Comparator:
    return request.app.state.comparator


ComparatorDep = Annotated[Comparator, Depends(get_comparator)]


def create_app(
    settings: Settings | None = None,
    make_encoder: Callable[[Settings], Encoder] = ClipEncoder.from_settings,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Собирает приложение. Параметры нужны тестам, чтобы подменить модель и сеть."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Модель грузится один раз при старте. Первый запуск скачивает её в кэш Hugging Face (~600 МБ).
        encoder = make_encoder(settings)
        async with make_http_client(settings.image_timeout_s, http_transport) as http:
            app.state.comparator = Comparator(encoder, settings, http)
            yield

    app = FastAPI(title="Item Compare API", version=__version__, lifespan=lifespan)

    @app.exception_handler(ImageFetchError)
    async def image_fetch_error(request: Request, exc: ImageFetchError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    async def health(comparator: ComparatorDep) -> dict[str, str]:
        return {"status": "ok", "model": comparator.encoder.name, "device": str(comparator.encoder.device)}

    @app.post("/v1/compare")
    async def compare(req: CompareRequest, comparator: ComparatorDep) -> CompareResult:
        """Похожесть двух товаров в процентах."""
        return await comparator.compare(req.item_a, req.item_b)

    @app.post("/v1/match")
    async def match(req: MatchRequest, comparator: ComparatorDep) -> MatchResponse:
        """Сравнивает товар со списком кандидатов и сортирует их по похожести (поиск дублей)."""
        return MatchResponse(matches=await comparator.match(req.query, req.candidates, req.top_k))

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run("item_compare.api:app", host=settings.host, port=settings.port)
