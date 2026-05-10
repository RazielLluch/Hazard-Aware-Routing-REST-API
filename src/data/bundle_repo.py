"""File-backed CRUD for named node bundles.

On-disk layout:
  data/node_bundles/<graph_id>/<name>.json

Bundles are scoped per-graph because node ids differ across graphs (a
bundle on ``la_trinidad_subgraph_n200`` cannot be applied to a different
graph). The ``name`` is constrained to a slug pattern at the schema layer
so it's safe as a filename on Windows + POSIX.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core import paths
from ..core.errors import NotFound


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_bundles(graph_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in paths.list_bundle_names(graph_id):
        try:
            out.append(read_bundle(graph_id, name))
        except (NotFound, OSError):
            continue
    return out


def read_bundle(graph_id: str, name: str) -> dict[str, Any]:
    p = paths.bundle_path(graph_id, name)
    if not p.exists():
        raise NotFound(f"Bundle not found: {graph_id}/{name}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_bundle(
    graph_id: str,
    *,
    name: str,
    depot: str,
    stops: list[str],
    description: str | None = None,
) -> dict[str, Any]:
    p = paths.bundle_path(graph_id, name)
    if p.exists():
        raise FileExistsError(f"Bundle already exists: {graph_id}/{name}")
    now = _now()
    record: dict[str, Any] = {
        "name": name,
        "graph_id": graph_id,
        "depot": depot,
        "stops": list(stops),
        "created_at": now,
        "updated_at": now,
        "description": description,
    }
    _write_atomic(p, record)
    return record


def update_bundle(
    graph_id: str,
    name: str,
    *,
    depot: str | None = None,
    stops: list[str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    record = read_bundle(graph_id, name)
    if depot is not None:
        record["depot"] = depot
    if stops is not None:
        record["stops"] = list(stops)
    if description is not None:
        record["description"] = description
    record["updated_at"] = _now()
    _write_atomic(paths.bundle_path(graph_id, name), record)
    return record


def delete_bundle(graph_id: str, name: str) -> None:
    p = paths.bundle_path(graph_id, name)
    if not p.exists():
        raise NotFound(f"Bundle not found: {graph_id}/{name}")
    p.unlink()


def _write_atomic(target: Path, record: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    os.replace(tmp, target)


__all__ = [
    "list_bundles",
    "read_bundle",
    "create_bundle",
    "update_bundle",
    "delete_bundle",
]
