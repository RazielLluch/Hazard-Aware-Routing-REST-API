from functools import lru_cache

import networkx as nx

from ..core import paths
from ..core.errors import NotFound
from ..schemas.common import LatLng
from ..schemas.graph import GraphInfo, GraphNode


# Placeholder activation thresholds pending integration with the harness's
# `_det` config files (stage_200_*_RI{N}_det.json). Edge is BLOCKED when
# combined_hazard >= threshold[ri]; lower threshold means more edges blocked.
_RI_THRESHOLDS: dict[int, float] = {1: 1.01, 2: 0.85, 3: 0.65, 4: 0.45, 5: 0.15}


@lru_cache(maxsize=4)
def load_graph(graph_id: str) -> nx.MultiDiGraph:
    path = paths.graph_path(graph_id)
    if not path.exists():
        raise NotFound(f"GraphML not found for graph_id={graph_id!r}: {path}")
    g = nx.read_graphml(str(path), force_multigraph=True)
    if not isinstance(g, nx.MultiDiGraph):
        g = nx.MultiDiGraph(g)
    return g


def get_graph_info(graph_id: str) -> GraphInfo:
    g = load_graph(graph_id)
    bbox = None
    xs: list[float] = []
    ys: list[float] = []
    for _, data in g.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))
    if xs and ys:
        bbox = (min(xs), min(ys), max(xs), max(ys))
    return GraphInfo(
        graph_id=graph_id,
        crs="EPSG:4326",
        num_nodes=g.number_of_nodes(),
        num_edges=g.number_of_edges(),
        bbox=bbox,
    )


def get_node(graph_id: str, node_id: str) -> GraphNode:
    g = load_graph(graph_id)
    if node_id not in g.nodes:
        raise NotFound(f"Node not found: {node_id!r}")
    return _node_from_graph(g, node_id)


def list_nodes(graph_id: str, ids: list[str] | None = None) -> list[GraphNode]:
    g = load_graph(graph_id)
    if ids is None:
        return [_node_from_graph(g, n) for n in g.nodes]
    return [_node_from_graph(g, n) for n in ids if n in g.nodes]


def _node_from_graph(g: nx.MultiDiGraph, node_id: str) -> GraphNode:
    data = g.nodes[node_id]
    return GraphNode(
        id=str(node_id),
        location=LatLng(
            lat=float(data.get("y", 0.0)),
            lng=float(data.get("x", 0.0)),
        ),
        street_count=data.get("street_count"),
        highway=data.get("highway"),
    )


@lru_cache(maxsize=8)
def scc_for_rain(graph_id: str, rain_level: int) -> tuple[str, ...]:
    g = load_graph(graph_id)
    activated = _activated_subgraph(g, rain_level)
    if activated.number_of_nodes() == 0:
        return ()
    components = list(nx.strongly_connected_components(activated))
    if not components:
        return ()
    largest = max(components, key=len)
    return tuple(str(n) for n in largest)


def _activated_subgraph(g: nx.MultiDiGraph, rain_level: int) -> nx.MultiDiGraph:
    threshold = _RI_THRESHOLDS.get(rain_level, 1.01)
    keep = [
        (u, v, k)
        for u, v, k, d in g.edges(keys=True, data=True)
        if float(d.get("combined_hazard", 0.0)) < threshold
    ]
    if not keep:
        return nx.MultiDiGraph()
    return g.edge_subgraph(keep).copy()
