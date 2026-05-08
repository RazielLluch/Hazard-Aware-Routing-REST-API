from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve(p: Path) -> Path:
    return p.expanduser().resolve()


AbsolutePath = Annotated[Path, AfterValidator(_resolve)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    harness_root: AbsolutePath = Field(
        default=Path("../../Benguet Flood and Landslide Data"),
    )
    graphs_root: AbsolutePath = Field(
        default=Path("../../Benguet Flood and Landslide Data/data"),
    )
    models_root: AbsolutePath = Field(
        default=Path("ml_models/latest"),
    )
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
    )
    log_level: str = Field(default="INFO")
    inference_device: str = Field(default="cpu")


settings = Settings()
