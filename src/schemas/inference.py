from typing import Annotated

from pydantic import Field

from .common import AlgorithmId, CamelModel, LatLng, Profile, RILevel
from .run import Run


class InferenceRequest(CamelModel):
    depot: str | LatLng
    delivery_stops: list[str | LatLng] = Field(min_length=1, max_length=20)
    rain_level: Annotated[int, Field(ge=1, le=5)]
    # Schema v3: prefer `algorithm` for explicit policy choice. NNA-* algos
    # are served live; DQN@* and `profile` short-circuit to 503 until the
    # finalised DQN model lands. `profile` is preserved for the legacy
    # /demo flow that was DQN-only.
    algorithm: AlgorithmId | None = None
    profile: Profile | None = None


class InferenceResponse(Run):
    model_version: str
    inference_ms: float
    # Schema v3: scenario context surfaced alongside the run so frontends
    # (notably /demo) can render BlockedEdgesLayer + synthesise a Scenario
    # for ScenarioPlaybackShell without a second fetch. Mirrors the subset
    # of Scenario fields the playback shell actually consumes.
    graph_id: str
    ri: RILevel
    start_node: str
    delivery_nodes: list[str] = Field(default_factory=list)
    blocked_edges: list[tuple[str, str]] = Field(default_factory=list)


class InferenceHealth(CamelModel):
    loaded: dict[Profile, list[int]] = Field(default_factory=dict)
    device: str = "cpu"
    is_warm: bool = False
