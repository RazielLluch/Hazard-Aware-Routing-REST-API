"""Live single-route inference.

Replaces the deterministic-mock stub with a real NNA execution path.
Loads the requested graph, builds a one-scenario ``Scenario`` via the
``deterministic_v3`` activation, runs the requested NNA policy through
``policy.run(scenario, view)``, and marshals the resulting ``Route`` into
an ``InferenceResponse``.

DQN inference is intentionally not yet wired — the request short-circuits
with a 503 ``DQNUnavailableError`` when ``profile`` is provided or when
``algorithm`` starts with ``DQN@``. When the teammate's finalised DQN
model lands, this is the swap-in point.
"""

from __future__ import annotations

import time
from typing import Optional

from ..core import paths
from ..core.errors import AppError, FeasibilityError
from ..data import graph_repo
from ..evaluation import run_policies
from ..evaluation.activation.deterministic_v3 import DeterministicV3Strategy
from ..evaluation.runners.base import build_graph_view
from ..evaluation.schemas import Scenario
from ..schemas.common import LatLng, Profile
from ..schemas.inference import InferenceHealth, InferenceRequest, InferenceResponse
from ..schemas.run import RouteEdge


_DEFAULT_ALGORITHM: str = "NNA-Dijkstra-HA"
_DEFAULT_DET_CONFIG_RELATIVE = (
    "src/evaluation/configs/hazard_training_final/balanced_HF/"
    "stage_200_balanced_HF_RI3_det.json"
)
_MAX_LIVE_STEPS: int = 220


class DQNUnavailableError(AppError):
    """503 - DQN inference requested but the underlying model is not yet wired.

    NNA family is served live by this endpoint; the teammate's DQN model
    swap-in lands here when ready.
    """

    code = "dqn_unavailable"
    status_code = 503


_activation_strategy: DeterministicV3Strategy | None = None


def _get_activation_strategy() -> DeterministicV3Strategy:
    """Cached `DeterministicV3Strategy` instance. The `_det` config's
    rain_levels and time-weight fields are profile-invariant, so one
    strategy is reused across all live inference calls.
    """
    global _activation_strategy
    if _activation_strategy is None:
        config_path = paths.project_root() / _DEFAULT_DET_CONFIG_RELATIVE
        _activation_strategy = DeterministicV3Strategy(config_path=config_path)
    return _activation_strategy


def route(graph_id: str, request: InferenceRequest) -> InferenceResponse:
    """Run a single live inference request through the requested NNA policy.

    Synchronous; the FastAPI router wraps this in `asyncio.to_thread` so
    the event loop stays free during the (typically sub-second) NNA run.
    """
    started = time.perf_counter()

    # Resolve algorithm; reject DQN until the model swap-in lands.
    algorithm = _resolve_algorithm(request)

    depot_node_id = _resolve_to_node_id(graph_id, request.depot)
    stop_node_ids = [_resolve_to_node_id(graph_id, s) for s in request.delivery_stops]

    # Build a one-shot scenario via deterministic_v3 activation.
    graph = graph_repo.load_graph(graph_id)
    ri_key = f"RI{request.rain_level}"
    strategy = _get_activation_strategy()
    blocked_edges = strategy.compute_blocked_edges(graph, ri_key)
    travel_time_map = strategy.compute_travel_time_map(graph, ri_key)

    scenario = Scenario(
        scenario_id="live",
        graph_id=graph_id,
        rain_level=int(request.rain_level),
        activation_mode=strategy.name,
        activation_seed=0,
        start_node=depot_node_id,
        delivery_nodes=stop_node_ids,
        blocked_edges=[list(e) for e in blocked_edges],
        travel_time_map=travel_time_map,
        max_steps=_MAX_LIVE_STEPS,
        num_deliveries=len(stop_node_ids),
        metadata={"source": "live_inference"},
    )

    view = build_graph_view(graph, scenario)
    policy = run_policies.POLICY_FACTORIES[algorithm]()
    result_route = policy.run(scenario, view)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    per_edge: list[RouteEdge] = [
        RouteEdge(
            step=int(e["step"]),
            u=str(e["u"]),
            v=str(e["v"]),
            length_m=float(e["length_m"]),
            travel_time=float(e["travel_time"]),
            hazard_flood=float(e["hazard_flood"]),
            hazard_landslide=float(e["hazard_landslide"]),
            was_replan=bool(e["was_replan"]),
            planned_next_edge=(
                tuple(e["planned_next_edge"])  # type: ignore[arg-type]
                if e.get("planned_next_edge")
                else None
            ),
        )
        for e in result_route.per_edge
    ]

    return InferenceResponse(
        scenario_id=result_route.scenario_id,
        algorithm_id=result_route.algorithm_id,  # type: ignore[arg-type]
        ri=ri_key,  # type: ignore[arg-type]
        success=result_route.success,
        failure_reason=result_route.failure_reason,
        replan_count=result_route.replan_count,
        wall_time_ms=result_route.wall_time_ms,
        visit_order=list(result_route.visit_order),
        edge_sequence=[(u, v) for u, v in result_route.edge_sequence],
        per_edge=per_edge,
        policy_metadata=dict(result_route.policy_metadata),
        algorithm_config_hash=result_route.algorithm_config_hash,
        model_version=f"v3-{algorithm}",
        inference_ms=elapsed_ms,
        graph_id=graph_id,
        start_node=depot_node_id,
        delivery_nodes=list(stop_node_ids),
        blocked_edges=[(u, v) for u, v in blocked_edges],
    )


def _resolve_algorithm(request: InferenceRequest) -> str:
    """Pick an algorithm from the request, rejecting DQN paths."""
    if request.profile is not None:
        raise DQNUnavailableError(
            f"DQN profile {request.profile!r} is not yet wired. "
            "Pass `algorithm` instead with one of the NNA-* algorithm ids."
        )
    algorithm = request.algorithm or _DEFAULT_ALGORITHM
    if algorithm.startswith("DQN@"):
        raise DQNUnavailableError(
            f"DQN algorithm {algorithm!r} is not yet wired."
        )
    if algorithm not in run_policies.POLICY_FACTORIES:
        raise FeasibilityError(
            f"Unknown algorithm {algorithm!r}. "
            f"Available: {sorted(run_policies.POLICY_FACTORIES)}"
        )
    return algorithm


def _resolve_to_node_id(graph_id: str, target: str | LatLng) -> str:
    if isinstance(target, str):
        graph_repo.get_node(graph_id, target)
        return target

    g = graph_repo.load_graph(graph_id)
    best_id: Optional[str] = None
    best_d2 = float("inf")
    for n, d in g.nodes(data=True):
        x = d.get("x")
        y = d.get("y")
        if x is None or y is None:
            continue
        dx = float(x) - target.lng
        dy = float(y) - target.lat
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_id = str(n)
    if best_id is None:
        raise FeasibilityError("No nodes available in graph for snapping.")
    return best_id


def health() -> InferenceHealth:
    """Inference availability snapshot.

    NNA runners are always available (torch-free). DQN profiles report as
    `[]` until the model swap-in lands; their `is_warm` flag tracks that.
    """
    loaded: dict[Profile, list[int]] = {"balanced": [], "fast": [], "safe": []}
    return InferenceHealth(
        loaded=loaded,
        device="cpu",
        is_warm=run_policies._HAS_TORCH,
    )
