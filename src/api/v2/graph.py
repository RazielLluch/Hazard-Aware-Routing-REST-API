"""GET /api/v2/graph — the full hazard graph + all (RI, profile) networks.

Cached in memory and on disk with an ETag; the frontend fetches it once per
session (the graph is invariant within a config). Building it cold is
expensive — it materialises all 15 (RI, profile) ``ActivatedNetwork`` instances.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

from fastapi import APIRouter, Header, Response

from ...core.config import settings
from ...macro.bridge import bridge
from .schemas import GraphExport

router = APIRouter(prefix="/graph", tags=["v2:graph"])

_cache: dict[str, object] = {}


def _build() -> tuple[str, bytes]:
    cached_body = _cache.get("body")
    if isinstance(cached_body, bytes):
        return str(_cache["etag"]), cached_body
    payload = bridge.graph_export()
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    etag = hashlib.sha256(body).hexdigest()[:32]
    cache_dir = settings.data_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "graph_export_v2.json").write_bytes(body)
    _cache["body"] = body
    _cache["etag"] = etag
    return etag, body


@router.get("", response_model=GraphExport, responses={304: {"description": "Not Modified — ETag matched"}})
def get_graph(
    if_none_match: Annotated[str | None, Header(alias="If-None-Match")] = None,
) -> Response:
    etag, body = _build()
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(
        content=body,
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": "public, max-age=3600"},
    )
