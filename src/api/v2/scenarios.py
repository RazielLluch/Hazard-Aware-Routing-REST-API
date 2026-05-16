"""GET /api/v2/scenarios[...] — paginated scenario manifests + precomputed / live runs."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Query

from ...core.errors import InferenceError, NotFound, ValidationFailure
from ...macro.bridge import bridge, resolve_algorithm
from . import artefacts
from .schemas import CompactResult, InferenceResponseV2, Page

router = APIRouter(prefix="/scenarios", tags=["v2:scenarios"])


@router.get("", response_model=Page)
def list_scenarios(
    cohort: Annotated[str, Query()] = "random",
    ri: Annotated[str | None, Query()] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500, alias="pageSize")] = 50,
) -> dict:
    """Paginated scenario manifests for a cohort, optionally filtered by RI key."""
    try:
        scenarios = artefacts.list_scenarios(cohort, ri)
    except KeyError as exc:
        raise ValidationFailure(f"Unknown cohort: {exc}") from exc
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc
    total = len(scenarios)
    start = (page - 1) * page_size
    return {
        "items": scenarios[start : start + page_size],
        "page": page,
        "page_size": page_size,
        "total": total,
    }


@router.get("/{scenario_id}/runs", response_model=CompactResult)
def get_run(
    scenario_id: str,
    profile: Annotated[str, Query()],
    algorithm: Annotated[str, Query()],
    cohort: Annotated[str, Query()] = "random",
) -> dict:
    """A precomputed trial row from the cohort's ``evaluation_raw.csv``.

    Returns 404 when the (scenario, profile, algorithm) combo is not precomputed
    — the frontend falls back to ``/runs/live`` on a 404.
    """
    try:
        spec = resolve_algorithm(algorithm)
    except ValueError as exc:
        raise ValidationFailure(str(exc)) from exc
    try:
        row = artefacts.get_precomputed_run(cohort, scenario_id, profile, spec)
    except KeyError as exc:
        raise ValidationFailure(f"Unknown cohort: {exc}") from exc
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc
    if row is None:
        raise NotFound(
            f"No precomputed run for {scenario_id} / {profile} / {algorithm} in cohort {cohort}."
        )
    return row


@router.get("/{scenario_id}/runs/live", response_model=InferenceResponseV2)
async def get_run_live(
    scenario_id: str,
    profile: Annotated[str, Query()],
    algorithm: Annotated[str, Query()],
    cohort: Annotated[str, Query()] = "random",
) -> dict:
    """Recompute a run live via the bridge — for combos absent from the precomputed CSVs."""
    try:
        scenario = artefacts.get_scenario(cohort, scenario_id)
    except KeyError as exc:
        raise ValidationFailure(f"Unknown cohort: {exc}") from exc
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc
    if scenario is None:
        raise NotFound(f"Scenario {scenario_id} not found in cohort {cohort}.")
    try:
        return await asyncio.to_thread(
            bridge.compute_route,
            algorithm=algorithm,
            profile=profile,
            ri=scenario["ri_key"],
            start=scenario["start"],
            deliveries=scenario["deliveries"],
        )
    except ValueError as exc:
        raise ValidationFailure(str(exc)) from exc
    except FileNotFoundError as exc:
        raise InferenceError(str(exc)) from exc
