from typing import Any

from pydantic import Field

from .common import AlgorithmId, CamelModel


class RouteEdge(CamelModel):
    step: int
    u: str
    v: str
    length_m: float = 0.0
    travel_time: float = 0.0
    hazard_flood: float = 0.0
    hazard_landslide: float = 0.0
    was_replan: bool = False


class RunSummary(CamelModel):
    scenario_id: str
    algorithm_id: AlgorithmId
    success: bool
    failure_reason: str | None = None
    replan_count: int = 0
    wall_time_ms: float = 0.0
    total_travel_time: float | None = None
    total_distance_m: float | None = None
    total_hazard_score: float | None = None


class Run(RunSummary):
    visit_order: list[str] = Field(default_factory=list)
    edge_sequence: list[tuple[str, str]] = Field(default_factory=list)
    per_edge: list[RouteEdge] = Field(default_factory=list)
    policy_metadata: dict[str, Any] = Field(default_factory=dict)
    algorithm_config_hash: str | None = None
