"""Node bundle CRUD scoped per graph.

  GET    /api/v1/graphs/{graph_id}/bundles
  GET    /api/v1/graphs/{graph_id}/bundles/{name}
  POST   /api/v1/graphs/{graph_id}/bundles
  PUT    /api/v1/graphs/{graph_id}/bundles/{name}
  DELETE /api/v1/graphs/{graph_id}/bundles/{name}

Mounted under ``/graphs/{graph_id}/bundles`` rather than a top-level
``/bundles`` route because bundles aren't reusable across graphs -- the
node ids differ.
"""

from fastapi import APIRouter, HTTPException, status

from ....data import bundle_repo
from ....schemas.bundle import (
    Bundle,
    BundleCreateRequest,
    BundleSummary,
    BundleUpdateRequest,
)

router = APIRouter(prefix="/graphs/{graph_id}/bundles", tags=["bundles"])


@router.get("", response_model=list[BundleSummary])
def list_bundles(graph_id: str) -> list[BundleSummary]:
    records = bundle_repo.list_bundles(graph_id)
    return [
        BundleSummary(
            name=r["name"],
            graph_id=r["graph_id"],
            num_stops=len(r.get("stops") or []),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in records
    ]


@router.get("/{name}", response_model=Bundle)
def get_bundle(graph_id: str, name: str) -> Bundle:
    return Bundle(**bundle_repo.read_bundle(graph_id, name))


@router.post("", response_model=Bundle, status_code=status.HTTP_201_CREATED)
def create_bundle(graph_id: str, request: BundleCreateRequest) -> Bundle:
    try:
        record = bundle_repo.create_bundle(
            graph_id,
            name=request.name,
            depot=request.depot,
            stops=request.stops,
            description=request.description,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return Bundle(**record)


@router.put("/{name}", response_model=Bundle)
def update_bundle(
    graph_id: str, name: str, request: BundleUpdateRequest
) -> Bundle:
    record = bundle_repo.update_bundle(
        graph_id,
        name,
        depot=request.depot,
        stops=request.stops,
        description=request.description,
    )
    return Bundle(**record)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_bundle(graph_id: str, name: str) -> None:
    bundle_repo.delete_bundle(graph_id, name)
