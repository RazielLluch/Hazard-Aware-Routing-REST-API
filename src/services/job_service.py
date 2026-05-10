"""High-level job operations: lifecycle transitions and SSE event streaming.

Composes ``data.job_repo`` (raw filesystem CRUD) into intent-named operations
the BenchmarkService and routers consume:

  * ``emit_event(job_id, stage, ...)`` -- append one progress line.
  * ``mark_running(...)`` / ``mark_succeeded(...)`` / ``mark_failed(...)`` --
    atomic status transitions on ``job.json``.
  * ``tail_events(job_id)`` -- async generator that yields existing events
    then polls ``events.jsonl`` for newly appended lines. Wired to a
    StreamingResponse in the jobs router for Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from ..data import job_repo


logger = logging.getLogger("services.job")


# Polling interval for the SSE tail loop. Short enough that progress events
# feel live; long enough that idle connections don't burn CPU.
_TAIL_POLL_SECONDS = 0.25

# Maximum tail duration before we close the SSE connection. Generous so
# slow benchmarks (DQN at 500 scenarios ~ 40s per profile + run_policies +
# evaluate) complete cleanly. Frontend can reconnect if it needs to keep
# watching.
_TAIL_MAX_SECONDS = 1800.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def emit_event(
    job_id: str,
    stage: str,
    *,
    current: int = 0,
    total: int = 0,
    message: str = "",
) -> None:
    job_repo.append_event(
        job_id,
        {
            "ts": _now(),
            "stage": stage,
            "current": current,
            "total": total,
            "message": message,
        },
    )


def mark_running(job_id: str, *, stage: str) -> None:
    job_repo.update_manifest(job_id, status="running", current_stage=stage)


def mark_succeeded(job_id: str, *, result_path: str) -> None:
    job_repo.update_manifest(
        job_id,
        status="succeeded",
        finished_at=_now(),
        result_path=result_path,
        error=None,
    )


def mark_failed(job_id: str, *, error: str) -> None:
    job_repo.update_manifest(
        job_id,
        status="failed",
        finished_at=_now(),
        error=error,
    )


async def tail_events(job_id: str) -> AsyncIterator[str]:
    """Yield SSE-formatted event lines.

    Strategy: replay all existing events.jsonl lines, then poll the file
    for new ones. Closes when the manifest reaches a terminal status
    (``succeeded`` or ``failed``) or when ``_TAIL_MAX_SECONDS`` elapses.
    """
    seen = 0
    deadline = asyncio.get_event_loop().time() + _TAIL_MAX_SECONDS

    while True:
        events = list(job_repo.stream_events(job_id))
        for ev in events[seen:]:
            yield _sse_format("progress", ev)
        seen = len(events)

        try:
            manifest = job_repo.read_manifest(job_id)
        except FileNotFoundError:
            yield _sse_format("error", {"message": f"job {job_id} not found"})
            return

        if manifest.get("status") in ("succeeded", "failed"):
            yield _sse_format("status", {
                "status": manifest["status"],
                "result_path": manifest.get("result_path"),
                "error": manifest.get("error"),
            })
            return

        if asyncio.get_event_loop().time() >= deadline:
            yield _sse_format("status", {
                "status": "timeout",
                "message": "SSE tail exceeded max duration; reconnect to keep watching.",
            })
            return

        await asyncio.sleep(_TAIL_POLL_SECONDS)


def _sse_format(event_name: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), sort_keys=True)
    return f"event: {event_name}\ndata: {payload}\n\n"


__all__ = [
    "emit_event",
    "mark_running",
    "mark_succeeded",
    "mark_failed",
    "tail_events",
]
