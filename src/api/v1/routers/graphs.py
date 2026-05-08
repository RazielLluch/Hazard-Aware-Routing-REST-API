from typing import Annotated

from fastapi import APIRouter, Query

from ....data import graph_repo
from ....schemas.graph import (
    GraphInfo,
    GraphNode,
    SampleNodesRequest,
    SampleNodesResponse,
)
from ....services import sampling_service

router = APIRouter(prefix="/graphs", tags=["graphs"])


@router.get("/{graph_id}", response_model=GraphInfo)
def get_graph(graph_id: str) -> GraphInfo:
    return graph_repo.get_graph_info(graph_id)


@router.get("/{graph_id}/nodes", response_model=list[GraphNode])
def list_nodes(
    graph_id: str,
    ids: Annotated[str | None, Query(description="Comma-separated node IDs")] = None,
) -> list[GraphNode]:
    id_list = [s.strip() for s in ids.split(",") if s.strip()] if ids else None
    return graph_repo.list_nodes(graph_id, id_list)


@router.post("/{graph_id}/sample-nodes", response_model=SampleNodesResponse)
def sample_nodes(graph_id: str, request: SampleNodesRequest) -> SampleNodesResponse:
    return sampling_service.sample_feasible_nodes(graph_id, request)
