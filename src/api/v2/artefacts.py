"""Read-only access to precomputed Macro-DDQN artefacts (scenarios + eval CSVs).

The runner writes per-cohort scenario manifests (``evaluation_scenarios*.jsonl``)
and per-trial results (``results*/evaluation_raw.csv``) under each artefact
directory. This module reads them for the v2 scenario / run endpoints — it does
not touch the runner or torch. Mirrors the helpers in
``macro_DDQN/visualization/export_viewer_data.py``.

Cross-artefact note: scenario manifests are read from the *default* artefact
(same eval seed -> identical scenarios across the non-GIS artefacts), but a
precomputed run is read from the artefact that actually trained that algorithm
(``AlgorithmSpec.artefact_dir``) — the greedy/oracle baselines appear in every
artefact's CSV, so they fall back to the default.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.macro.bridge import AlgorithmSpec

# cohort -> (scenario manifest filename, results CSV relative path, label)
_COHORTS: dict[str, tuple[str, str, str]] = {
    "random": (
        "evaluation_scenarios.jsonl",
        "results/evaluation_raw.csv",
        "Random",
    ),
    "hazard_opportunity": (
        "evaluation_scenarios_hazard_opportunity.jsonl",
        "results_hazard_opportunity/evaluation_raw.csv",
        "Hazard Opportunity",
    ),
    "risk_time_tradeoff": (
        "evaluation_scenarios_risk_time_tradeoff.jsonl",
        "results_risk_time_tradeoff/evaluation_raw.csv",
        "Risk-Time Tradeoff",
    ),
}


def _artifacts_root() -> Path:
    return Path(settings.macro_vendor_root) / "macro_DDQN" / "artifacts"


def _default_artefact_dir() -> Path:
    return _artifacts_root() / settings.macro_default_artefact


def _fnum(value: Any) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compact_result(row: dict[str, str]) -> dict[str, Any]:
    """One ``evaluation_raw.csv`` row -> the v2 ``CompactResult`` shape.

    ``success`` is intentionally dropped — under soft-blocking it is always 1.
    """
    return {
        "scenario_id": row["scenario_id"],
        "ri_key": row["ri_key"],
        "profile": row["profile"],
        "algorithm": row["algorithm"],
        "delivered": int(row["delivered"]),
        "objective": _fnum(row["objective"]),
        "time_min": _fnum(row["travel_time_min"]),
        "distance_km": round(_fnum(row["distance_m"]) / 1000.0, 6),
        "flood": _fnum(row["flood_exposure"]),
        "landslide": _fnum(row["landslide_exposure"]),
        "blockage": _fnum(row["blockage_exposure"]),
        "common_risk": _fnum(row["common_risk_exposure"]),
        "hops": int(row["hops"]),
        "visit_order": row["visit_order"].split("|") if row["visit_order"] else [],
    }


def list_scenario_sets() -> list[dict[str, Any]]:
    """One ``ScenarioSetSummary`` per cohort whose manifest exists on disk."""
    artefact = _default_artefact_dir()
    summaries: list[dict[str, Any]] = []
    for cohort, (scn_file, _csv_file, label) in _COHORTS.items():
        scn_path = artefact / scn_file
        if not scn_path.exists():
            continue
        counts: dict[str, int] = {}
        for scn in _read_jsonl(scn_path):
            ri = str(scn["ri_key"])
            counts[ri] = counts.get(ri, 0) + 1
        summaries.append(
            {
                "cohort": cohort,
                "label": label,
                "counts_by_ri": counts,
                "total": sum(counts.values()),
                "artefact_id": settings.macro_default_artefact,
            }
        )
    return summaries


def list_scenarios(cohort: str, ri: str | None = None) -> list[dict[str, Any]]:
    """Scenario manifests for a cohort, optionally filtered by RI key."""
    if cohort not in _COHORTS:
        raise KeyError(cohort)
    scn_file, _csv_file, _label = _COHORTS[cohort]
    scn_path = _default_artefact_dir() / scn_file
    if not scn_path.exists():
        raise FileNotFoundError(f"Scenario manifest not found: {scn_path}")
    out: list[dict[str, Any]] = []
    for scn in _read_jsonl(scn_path):
        if ri and str(scn["ri_key"]) != ri:
            continue
        out.append(
            {
                "scenario_id": str(scn["scenario_id"]),
                "ri_key": str(scn["ri_key"]),
                "start": str(scn["start"]),
                "deliveries": [str(node) for node in scn["deliveries"]],
            }
        )
    return out


def get_scenario(cohort: str, scenario_id: str) -> dict[str, Any] | None:
    """Look up a single scenario manifest by id (linear scan — fine at this scale)."""
    for scenario in list_scenarios(cohort):
        if scenario["scenario_id"] == scenario_id:
            return scenario
    return None


def get_precomputed_run(
    cohort: str, scenario_id: str, profile: str, spec: AlgorithmSpec
) -> dict[str, Any] | None:
    """Read one precomputed trial row from the right artefact's ``evaluation_raw.csv``.

    Learned variants are read from their own ``spec.artefact_dir``; greedy /
    oracle baselines (``artefact_dir is None``) appear in every artefact's CSV,
    so they fall back to the default artefact. The CSV's ``algorithm`` column
    holds the runner's collapsed label, so we filter on ``spec.runner_algorithm``
    and then relabel the row with the caller-facing ``spec.id``.
    """
    if cohort not in _COHORTS:
        raise KeyError(cohort)
    _scn_file, csv_file, _label = _COHORTS[cohort]
    artefact_name = spec.artefact_dir or settings.macro_default_artefact
    csv_path = _artifacts_root() / artefact_name / csv_file
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    for row in _read_csv(csv_path):
        if (
            row["scenario_id"] == scenario_id
            and row["profile"] == profile
            and row["algorithm"] == spec.runner_algorithm
        ):
            result = compact_result(row)
            result["algorithm"] = spec.id  # relabel runner column -> v2 catalog id
            return result
    return None


def iter_metric_rows(cohort: str) -> list[dict[str, Any]]:
    """All trial rows for all 7 algorithms in a cohort, relabelled to v2 ids.

    Each learned variant is read from its own ``spec.artefact_dir``; the greedy
    and oracle baselines come from the default artefact (they appear in every
    artefact's CSV). Returns flat dicts with just the fields the
    ``/api/v2/metrics`` aggregation needs.
    """
    from src.macro.bridge import _ALGORITHM_SPECS

    if cohort not in _COHORTS:
        raise KeyError(cohort)
    _scn_file, csv_file, _label = _COHORTS[cohort]
    csv_cache: dict[Path, list[dict[str, str]]] = {}
    rows: list[dict[str, Any]] = []
    for spec in _ALGORITHM_SPECS:
        artefact_name = spec.artefact_dir or settings.macro_default_artefact
        csv_path = _artifacts_root() / artefact_name / csv_file
        if not csv_path.exists():
            continue
        if csv_path not in csv_cache:
            csv_cache[csv_path] = _read_csv(csv_path)
        for row in csv_cache[csv_path]:
            if row["algorithm"] != spec.runner_algorithm:
                continue
            distance_m = _fnum(row["distance_m"])
            rows.append(
                {
                    "cohort": cohort,
                    "algorithm": spec.id,
                    "category": spec.category,
                    "profile": row["profile"],
                    "ri_key": row["ri_key"],
                    "objective": _fnum(row["objective"]),
                    "travel_time_min": _fnum(row["travel_time_min"]),
                    "distance_m": distance_m,
                    "distance_km": round(distance_m / 1000.0, 6),
                    "common_risk_exposure": _fnum(row["common_risk_exposure"]),
                    "blockage_exposure": _fnum(row["blockage_exposure"]),
                    "hops": int(row["hops"]),
                    "replan_count": int(row.get("replan_count", 0) or 0),
                }
            )
    return rows
