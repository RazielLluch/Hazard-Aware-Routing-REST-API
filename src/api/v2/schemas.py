"""Pydantic request/response models for the ``/api/v2`` namespace.

Unlike the v1 schemas (which use ``CamelModel`` aliasing), v2 models are plain
snake_case ``BaseModel``s — they mirror the JSON produced by
``src.macro.bridge`` and ``macro_DDQN/visualization/export_viewer_data.py``
1:1, so no field aliasing is needed. ``success`` is intentionally absent
everywhere: under soft-blocking it is always true (see the implementation
reference, "What success means now").
"""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Inference (POST /api/v2/inference, GET /scenarios/{id}/runs/live)
# ---------------------------------------------------------------------------
class InferenceRequestV2(BaseModel):
    algorithm: str
    profile: str
    ri_key: str
    start: str
    deliveries: list[str]
    max_steps: int = 1000


class NodePayload(BaseModel):
    id: str
    x: float
    y: float
    lng: float
    lat: float


class RouteMetrics(BaseModel):
    objective: float
    time_min: float
    distance_m: float
    distance_km: float
    flood_exposure: float
    landslide_exposure: float
    blockage_exposure: float
    risk_exposure: float
    common_risk_exposure: float
    hops: int


class RouteSegment(BaseModel):
    u: str
    v: str
    coordinates: list[NodePayload]
    length_m: float
    base_time_min: float
    travel_time_min: float
    objective: float
    flood_hazard: float
    landslide_hazard: float
    flood_active: int
    landslide_active: int
    flood_blocked: int
    landslide_blocked: int
    blocked: int
    flood_exposure: float
    landslide_exposure: float
    blockage_exposure: float
    risk_exposure: float
    common_risk_exposure: float


class RouteLeg(BaseModel):
    source: str
    target: str
    path_nodes: list[str]
    metrics: RouteMetrics
    segments: list[RouteSegment]


class InferenceResponseV2(BaseModel):
    algorithm: str
    profile: str
    ri_key: str
    artefact_id: str | None
    delivered: int
    failure_reason: str
    start: NodePayload
    deliveries: list[NodePayload]
    visit_order: list[str]
    metrics: RouteMetrics
    legs: list[RouteLeg]


# ---------------------------------------------------------------------------
# Algorithm catalog (GET /api/v2/algorithms)
# ---------------------------------------------------------------------------
class AlgorithmEntry(BaseModel):
    id: str
    label: str
    category: str  # "learned" | "baseline" | "oracle"
    requires_model: bool
    aliases: list[str]


# ---------------------------------------------------------------------------
# Graph export (GET /api/v2/graph)
# ---------------------------------------------------------------------------
class GraphExportNode(BaseModel):
    id: str
    x: float
    y: float


class NetworkEdge(BaseModel):
    u: str
    v: str
    base_time: float
    travel_time: float
    objective: float
    length_m: float
    flood: float
    landslide: float
    combined: float  # max(flood, landslide)
    flood_active: int
    landslide_active: int
    flood_blocked: int
    landslide_blocked: int
    blocked: int


class GraphExport(BaseModel):
    nodes: list[GraphExportNode]
    networks: dict[str, list[NetworkEdge]]  # key = "<RI>:<profile>"
    profiles: list[str]
    ri_keys: list[str]
    algorithms: list[str]
    artefact_id: str | None = None


# ---------------------------------------------------------------------------
# Scenario sets + scenarios + precomputed runs
# ---------------------------------------------------------------------------
class ScenarioSetScenario(BaseModel):
    scenario_id: str
    ri_key: str
    start: str
    deliveries: list[str]


class ScenarioSetSummary(BaseModel):
    cohort: str  # "random" | "hazard_opportunity" | "risk_time_tradeoff"
    label: str
    counts_by_ri: dict[str, int]
    total: int
    artefact_id: str | None = None


class CompactResult(BaseModel):
    scenario_id: str
    ri_key: str
    profile: str
    algorithm: str
    # `success` intentionally dropped for v2 — always true under soft-blocking.
    delivered: int
    objective: float
    time_min: float
    distance_km: float
    flood: float
    landslide: float
    blockage: float
    common_risk: float
    hops: int
    visit_order: list[str]


class Page(BaseModel):
    items: list[ScenarioSetScenario]
    page: int
    page_size: int
    total: int


class HealthV2(BaseModel):
    loaded: bool
    config: str | None = None
    feature_count: int = 0
    cached_networks: int = 0
    cached_models: int = 0
