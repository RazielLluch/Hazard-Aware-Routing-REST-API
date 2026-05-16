"""GET /api/v2/metrics — manuscript-facing aggregates over the precomputed cohorts.

Facets the per-trial evaluation rows by ``(cohort, profile, ri_key, algorithm)``
and means the manuscript metrics. ``success_rate`` is intentionally absent — it
is always 1.0 under soft-blocking. Also emits the baseline-reduction view used
in the manuscript: ``100 * (mean_baseline - mean_model) / mean_baseline``.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ...core.errors import NotFound, ValidationFailure
from . import artefacts

router = APIRouter(prefix="/metrics", tags=["v2:metrics"])

# Metrics meaned per facet cell.
_METRIC_FIELDS = (
    "objective",
    "travel_time_min",
    "distance_km",
    "common_risk_exposure",
    "blockage_exposure",
    "hops",
    "replan_count",
)
# Metrics carried in the baseline-reduction view (lower-is-better quantities).
_REDUCTION_METRICS = ("common_risk_exposure", "travel_time_min", "objective", "blockage_exposure")
_LEARNED = ("Macro-DQN", "Macro-DDQN-Online-Scratch", "Macro-DDQN-Offline-NoAnchor")
_COMPARATORS = (
    "Greedy-Time-Base",
    "Greedy-Time-SoftPenalty",
    "Greedy-HazardAware",
    "Macro-DP-Oracle",
)


class MetricFacet(BaseModel):
    cohort: str
    profile: str
    ri_key: str
    algorithm: str
    category: str
    n: int
    objective: float
    travel_time_min: float
    distance_km: float
    common_risk_exposure: float
    blockage_exposure: float
    hops: float
    replan_count: float


class BaselineReduction(BaseModel):
    profile: str
    ri_key: str
    model: str  # the learned algorithm
    baseline: str  # the comparator (greedy baseline or the oracle)
    metric: str
    mean_model: float
    mean_baseline: float
    reduction_pct: float  # 100 * (mean_baseline - mean_model) / mean_baseline


class MetricsResponse(BaseModel):
    cohort: str
    facets: list[MetricFacet]
    baseline_reductions: list[BaselineReduction]


def _aggregate(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, float]]]:
    """Group rows by (profile, ri_key, algorithm) and mean the metric fields."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["profile"], row["ri_key"], row["algorithm"])].append(row)

    facets: list[dict[str, Any]] = []
    means: dict[tuple[str, str, str], dict[str, float]] = {}
    for (profile, ri_key, algorithm), group in sorted(groups.items()):
        cell_means = {
            field: round(statistics.fmean(r[field] for r in group), 6)
            for field in _METRIC_FIELDS
        }
        means[(profile, ri_key, algorithm)] = cell_means
        facets.append(
            {
                "cohort": group[0]["cohort"],
                "profile": profile,
                "ri_key": ri_key,
                "algorithm": algorithm,
                "category": group[0]["category"],
                "n": len(group),
                **cell_means,
            }
        )
    return facets, means


def _baseline_reductions(
    means: dict[tuple[str, str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    """For each learned algorithm, percent reduction vs each comparator per (profile, RI)."""
    out: list[dict[str, Any]] = []
    for (profile, ri_key, algorithm), model_means in means.items():
        if algorithm not in _LEARNED:
            continue
        for comparator in _COMPARATORS:
            base_means = means.get((profile, ri_key, comparator))
            if base_means is None:
                continue
            for metric in _REDUCTION_METRICS:
                mean_model = model_means[metric]
                mean_baseline = base_means[metric]
                reduction_pct = (
                    round(100.0 * (mean_baseline - mean_model) / mean_baseline, 4)
                    if mean_baseline
                    else 0.0
                )
                out.append(
                    {
                        "profile": profile,
                        "ri_key": ri_key,
                        "model": algorithm,
                        "baseline": comparator,
                        "metric": metric,
                        "mean_model": mean_model,
                        "mean_baseline": mean_baseline,
                        "reduction_pct": reduction_pct,
                    }
                )
    return out


@router.get("", response_model=MetricsResponse)
def get_metrics(cohort: Annotated[str, Query()] = "random") -> dict:
    """Faceted metric means + the manuscript baseline-reduction view for one cohort."""
    try:
        rows = artefacts.iter_metric_rows(cohort)
    except KeyError as exc:
        raise ValidationFailure(f"Unknown cohort: {exc}") from exc
    if not rows:
        raise NotFound(f"No evaluation results found for cohort {cohort!r}.")
    facets, means = _aggregate(rows)
    return {
        "cohort": cohort,
        "facets": facets,
        "baseline_reductions": _baseline_reductions(means),
    }
