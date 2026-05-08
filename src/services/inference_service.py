import time

from ..core.errors import FeasibilityError
from ..data import graph_repo
from ..schemas.common import LatLng
from ..schemas.inference import InferenceHealth, InferenceRequest, InferenceResponse
from ..schemas.run import RouteEdge


def route(graph_id: str, request: InferenceRequest) -> InferenceResponse:
    started = time.perf_counter()

    depot_node_id = _resolve_to_node_id(graph_id, request.depot)
    stop_node_ids = [_resolve_to_node_id(graph_id, s) for s in request.delivery_stops]
    visit_order = [depot_node_id, *stop_node_ids]

    edge_sequence: list[tuple[str, str]] = []
    per_edge: list[RouteEdge] = []
    for i in range(len(visit_order) - 1):
        u, v = visit_order[i], visit_order[i + 1]
        edge_sequence.append((u, v))
        per_edge.append(
            RouteEdge(
                step=i,
                u=u,
                v=v,
                length_m=100.0,
                travel_time=2.0,
                hazard_flood=0.1,
                hazard_landslide=0.05,
                was_replan=False,
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    algo_id = f"DQN@{request.profile}_HF"

    return InferenceResponse(
        scenario_id="live",
        algorithm_id=algo_id,  # type: ignore[arg-type]
        success=True,
        failure_reason=None,
        replan_count=0,
        wall_time_ms=elapsed_ms,
        visit_order=visit_order,
        edge_sequence=edge_sequence,
        per_edge=per_edge,
        policy_metadata={"variant": "p0_stub", "profile": request.profile},
        algorithm_config_hash=None,
        model_version="p0-mock-v0",
        inference_ms=elapsed_ms,
    )


def _resolve_to_node_id(graph_id: str, target: str | LatLng) -> str:
    if isinstance(target, str):
        graph_repo.get_node(graph_id, target)
        return target

    g = graph_repo.load_graph(graph_id)
    best_id: str | None = None
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
    return InferenceHealth(
        loaded={"balanced": [], "fast": [], "safe": []},
        device="cpu",
        is_warm=False,
    )
