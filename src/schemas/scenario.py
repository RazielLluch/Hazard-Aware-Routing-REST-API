from typing import Annotated

from pydantic import Field

from .common import CamelModel, RILevel


class ScenarioListItem(CamelModel):
    scenario_id: str
    rain_level: Annotated[int, Field(ge=1, le=5)]
    ri: RILevel
    num_deliveries: int
    num_blocked_edges: int


class Scenario(CamelModel):
    scenario_id: str
    benchmark_id: str
    graph_id: str
    rain_level: Annotated[int, Field(ge=1, le=5)]
    ri: RILevel
    start_node: str
    delivery_nodes: list[str] = Field(default_factory=list)
    blocked_edges: list[tuple[str, str]] = Field(default_factory=list)
    max_steps: int = 220
    activation_mode: str = "deterministic_v3"
    activation_seed: int | None = None
