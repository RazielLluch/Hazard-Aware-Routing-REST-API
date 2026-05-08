from typing import Annotated

from pydantic import Field

from .common import CamelModel, LatLng


class GraphInfo(CamelModel):
    graph_id: str
    crs: str = "EPSG:4326"
    num_nodes: int
    num_edges: int
    bbox: tuple[float, float, float, float] | None = None
    source: str = "OSM via OSMnx"


class GraphNode(CamelModel):
    id: str
    location: LatLng
    street_count: int | None = None
    highway: str | None = None


class GraphEdge(CamelModel):
    u: str
    v: str
    length: float
    travel_time_min: float = 0.0
    flood_hazard: float = 0.0
    landslide_hazard: float = 0.0
    combined_hazard: float = 0.0
    geometry: list[LatLng] = Field(default_factory=list)
    highway: str | None = None
    name: str | None = None


class SampleNodesRequest(CamelModel):
    rain_level: Annotated[int, Field(ge=1, le=5)]
    k: Annotated[int, Field(ge=1, le=50)]
    seed: int | None = None


class SampleNodesResponse(CamelModel):
    feasible: bool
    scc_size: int
    seed: int
    depot: GraphNode | None = None
    nodes: list[GraphNode] = Field(default_factory=list)
