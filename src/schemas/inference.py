from typing import Annotated

from pydantic import Field

from .common import CamelModel, LatLng, Profile
from .run import Run


class InferenceRequest(CamelModel):
    depot: str | LatLng
    delivery_stops: list[str | LatLng] = Field(min_length=1, max_length=20)
    rain_level: Annotated[int, Field(ge=1, le=5)]
    profile: Profile


class InferenceResponse(Run):
    model_version: str
    inference_ms: float


class InferenceHealth(CamelModel):
    loaded: dict[Profile, list[int]] = Field(default_factory=dict)
    device: str = "cpu"
    is_warm: bool = False
