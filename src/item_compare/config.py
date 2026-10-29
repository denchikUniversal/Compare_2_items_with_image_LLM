from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки сервиса.

    Любое поле переопределяется переменной окружения с префиксом ITEM_COMPARE_
    (например, ITEM_COMPARE_DEVICE=cpu) или строкой в файле .env.
    """

    model_config = SettingsConfigDict(env_prefix="ITEM_COMPARE_", env_file=".env", extra="ignore")

    # Модель с общим пространством для картинок и текста (только английский текст).
    model_name: str = "openai/clip-vit-base-patch32"
    # auto | cuda | mps | cpu
    device: str = "auto"
    # Сколько картинок/текстов прогонять через модель за раз (ограничивает память).
    batch_size: int = 32

    # Загрузка картинок по ссылке.
    image_timeout_s: float = 10.0
    image_max_bytes: int = 10 * 1024 * 1024

    # Вклад картинки и текста в итоговый процент, когда у обоих товаров есть и то, и другое.
    weight_image: float = 0.6
    weight_text: float = 0.4
    # С какого итогового процента считаем, что это один и тот же товар.
    same_threshold: float = 50.0

    # Результаты обучения (см. item_compare.training). Пусто — работают значения по умолчанию.
    calibration_path: Path | None = None
    head_path: Path | None = None

    host: str = "127.0.0.1"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
