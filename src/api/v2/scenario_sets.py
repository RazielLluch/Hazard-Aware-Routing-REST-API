"""GET /api/v2/scenario_sets — available precomputed cohorts + per-RI counts."""

from __future__ import annotations

from fastapi import APIRouter

from . import artefacts
from .schemas import ScenarioSetSummary

router = APIRouter(prefix="/scenario_sets", tags=["v2:scenario_sets"])


@router.get("", response_model=list[ScenarioSetSummary])
def list_scenario_sets() -> list[dict]:
    """One summary per cohort manifest present on disk.

    Cohorts: ``random`` / ``hazard_opportunity`` / ``risk_time_tradeoff``.
    """
    return artefacts.list_scenario_sets()
