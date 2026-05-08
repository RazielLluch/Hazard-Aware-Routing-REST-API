from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


T = TypeVar("T")


RILevel = Literal["RI1", "RI2", "RI3", "RI4", "RI5"]
RILevels: tuple[RILevel, ...] = ("RI1", "RI2", "RI3", "RI4", "RI5")

AlgorithmId = Literal[
    "NNA-Dijkstra",
    "NNA-AStar",
    "NNA-Dijkstra-HA",
    "NNA-Dijkstra-Blind",
    "NNA-AStar-Blind",
    "NNA-Dijkstra-HA-Blind",
    "DQN@balanced_HF",
    "DQN@fast_HF",
    "DQN@safe_HF",
]

Profile = Literal["balanced", "fast", "safe"]


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class LatLng(CamelModel):
    lat: Annotated[float, Field(ge=-90, le=90)]
    lng: Annotated[float, Field(ge=-180, le=180)]


class Page(CamelModel, Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total: int


class ErrorEnvelope(CamelModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


def ri_from_int(n: int) -> RILevel:
    if not 1 <= n <= 5:
        raise ValueError(f"rain_level must be 1..5, got {n}")
    return f"RI{n}"  # type: ignore[return-value]


def int_from_ri(ri: RILevel) -> int:
    return int(ri[2:])
