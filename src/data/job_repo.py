"""File-backed CRUD for benchmark-generation jobs.

On-disk layout (rename plan §0.1):
  data/jobs/<job_id>/
    job.json        manifest (rewritten on each status transition)
    events.jsonl    append-only progress feed (one JSON object per line)

Mirrors the BenchmarkRepo pattern: pure functions over filesystem state,
no in-process registry. Survives backend restart trivially.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core import paths
from ..core.errors import NotFound
from .jsonl import stream_jsonl


def new_job_id(prefix: str = "job") -> str:
    """Generate a sortable, human-readable job id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{suffix}"


def create_job(job_id: str, kind: str, config: dict[str, Any]) -> dict[str, Any]:
    """Initialise a job. Writes ``job.json`` with ``status='queued'`` and
    creates an empty ``events.jsonl``.
    """
    jdir = paths.job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "config": config,
        "started_at": started_at,
        "finished_at": None,
        "current_stage": None,
        "result_path": None,
        "error": None,
    }
    _write_manifest_atomic(job_id, manifest)
    paths.job_events_path(job_id).touch(exist_ok=True)
    return manifest


def read_manifest(job_id: str) -> dict[str, Any]:
    p = paths.job_manifest_path(job_id)
    if not p.exists():
        raise NotFound(f"Job not found: {job_id!r}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_manifest(job_id: str, **patch: Any) -> dict[str, Any]:
    """Merge ``patch`` into ``job.json`` and atomically rewrite it."""
    manifest = read_manifest(job_id)
    manifest.update(patch)
    _write_manifest_atomic(job_id, manifest)
    return manifest


def append_event(job_id: str, event: dict[str, Any]) -> None:
    """Append one JSON line to ``events.jsonl``. Single-writer; no lock
    needed because the BenchmarkService runs jobs sequentially.
    """
    p = paths.job_events_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(",", ":"), sort_keys=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def list_summaries() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for job_id in paths.list_known_job_ids():
        try:
            manifest = read_manifest(job_id)
        except (NotFound, OSError):
            continue
        out.append({k: manifest.get(k) for k in (
            "job_id",
            "kind",
            "status",
            "started_at",
            "finished_at",
            "current_stage",
            "result_path",
        )})
    return out


def stream_events(job_id: str) -> Iterator[dict[str, Any]]:
    """Yield each event currently in ``events.jsonl``. Caller is responsible
    for tailing additional lines as they're appended.
    """
    p = paths.job_events_path(job_id)
    if not p.exists():
        return
    yield from stream_jsonl(p)


def _write_manifest_atomic(job_id: str, manifest: dict[str, Any]) -> None:
    """Atomic rewrite: write to ``<name>.tmp`` then ``os.replace``.

    On Windows, ``os.replace`` is atomic for files within the same filesystem,
    so partial reads by an SSE consumer never see a half-written manifest.
    """
    target = paths.job_manifest_path(job_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, target)


__all__ = [
    "new_job_id",
    "create_job",
    "read_manifest",
    "update_manifest",
    "append_event",
    "list_summaries",
    "stream_events",
]
