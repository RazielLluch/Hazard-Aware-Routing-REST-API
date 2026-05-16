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

    data_root: AbsolutePath = Field(default=Path("data"))
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
    )
    log_level: str = Field(default="INFO")
    inference_device: str = Field(default="cpu")

    # Macro-DDQN bridge (src/macro/). Mirrors the upstream layout
    # (``macro_DDQN/`` + ``map_preprocessing/``) inside the REST API folder so
    # the vendored runner's ``resolve_path()`` works with no config mutations.
    # The vendored runner reads this via the ``MACRO_DDQN_ROOT`` env var to
    # resolve the hazard graph, configs, and artefacts.
    macro_vendor_root: AbsolutePath = Field(default=Path("data/macro_vendor"))
    macro_default_config: str = Field(
        default="final_macro_dqn_v2_25yr_adjusted_s500_e300_with_cohorts_offline_ddqn_no_anchor.json"
    )
    macro_default_artefact: str = Field(
        default="final_macro_dqn_v2_25yr_adjusted_s500_e300_with_cohorts_offline_ddqn_no_anchor"
    )


settings = Settings()
