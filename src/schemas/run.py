from typing import Any

from pydantic import Field

from .common import AlgorithmId, CamelModel, RILevel


class RouteEdge(CamelModel):
    step: int
    u: str
    v: str
    length_m: float = 0.0
    travel_time: float = 0.0
    hazard_flood: float = 0.0
    hazard_landslide: float = 0.0
    was_replan: bool = False
    # Schema v3: when was_replan is true, this is the (u, v) edge the
    # planner attempted before the block forced a replan. Optional for
    # backward compat; legacy v2 routes have no such field.
    planned_next_edge: tuple[str, str] | None = None


class RunSummary(CamelModel):
    scenario_id: str
    algorithm_id: AlgorithmId
    # Schema v3: surfaced from the parent scenario so the frontend can
    # group a flat list of run summaries by RI without a second fetch.
    # Optional only because the join may fail during backwards-compat
    # reads against legacy v2 cohorts that lack a scenarios.jsonl entry.
    ri: RILevel | None = None
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
