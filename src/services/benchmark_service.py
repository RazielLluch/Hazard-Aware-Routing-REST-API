"""Benchmark generation orchestrator.

Calls the vendored evaluation pipeline (scenario_generator, run_policies,
evaluator) in-process, with each CPU-bound stage wrapped in
``asyncio.to_thread`` so the FastAPI event loop stays responsive to /jobs
status polls during a benchmark run.

Emits stage-level progress events to ``data/jobs/<job_id>/events.jsonl``
via ``job_service``. Stage-level granularity matches the SSE consumer's
three-stage progress UI; per-RI/per-algorithm events would need a
callback threaded through the harness.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..core import paths
from ..core.errors import NotFound
from ..data import bundle_repo, job_repo
from ..evaluation import evaluator, run_policies, scenario_generator
from . import job_service


logger = logging.getLogger("services.benchmark")


# Any one `_det` config works for scenario generation -- the rain_levels
# and time-weight fields are shared across profiles. We pick balanced as
# the canonical default.
_DEFAULT_CONFIG = (
    Path("src") / "evaluation" / "configs" / "hazard_training_final"
    / "balanced_HF" / "stage_200_balanced_HF_RI3_det.json"
)


def _resolve_graph_path(graph_id: str) -> Path:
    """Return the absolute graphml path for a known graph_id."""
    try:
        return paths.graph_path(graph_id)
    except ValueError as exc:
        raise ValueError(f"Unknown graph_id: {graph_id!r}") from exc


def _validate_algorithms(algorithms: list[str]) -> None:
    """Reject DQN algorithms when torch is unavailable; reject empty
    algorithm lists; reject unknown algorithm ids.
    """
    if not algorithms:
        raise ValueError("at least one algorithm must be specified")
    unknown = [a for a in algorithms if a not in run_policies.POLICY_FACTORIES]
    if unknown:
        # If any unknowns are DQN-shaped and torch is missing, give a
        # specific error code that the API surface can short-circuit on.
        dqn_unavailable = [a for a in unknown if a.startswith("DQN@")]
        if dqn_unavailable and not run_policies._HAS_TORCH:
            raise DQNUnavailableError(
                f"DQN algorithms requested but torch is not installed: {sorted(dqn_unavailable)}"
            )
        raise ValueError(
            f"Unknown algorithm(s) {sorted(unknown)}; "
            f"available: {sorted(run_policies.POLICY_FACTORIES)}"
        )


class DQNUnavailableError(RuntimeError):
    """Raised when a benchmark requests DQN runners but torch is absent."""


async def run_benchmark_job(job_id: str, config: dict[str, Any]) -> None:
    """Execute the full pipeline for one benchmark generation job.

    Designed to run in a FastAPI ``BackgroundTasks`` worker. Errors are
    captured and written to ``job.json`` so the SSE consumer can surface
    them; this function never raises.
    """
    try:
        await _run_pipeline(job_id, config)
    except Exception as exc:  # noqa: BLE001 -- intentional catch-all at boundary
        logger.exception("benchmark job %s failed", job_id)
        job_service.emit_event(job_id, "evaluate", message=f"job failed: {exc}")
        job_service.mark_failed(job_id, error=str(exc))


async def _run_pipeline(job_id: str, config: dict[str, Any]) -> None:
    benchmark_id = config["benchmark_id"]
    graph_id = config["graph_id"]
    graph_path = _resolve_graph_path(graph_id)

    benchmark_dir = paths.benchmark_dir(benchmark_id)
    project_root = paths.project_root()
    config_path = project_root / _DEFAULT_CONFIG

    algorithms = config.get("algorithms") or []
    _validate_algorithms(algorithms)

    # Optional: resolve a saved bundle into inject_depot / inject_stops
    # kwargs. Validates up-front so a missing or wrong-graph bundle fails
    # the job before any scenario generation work happens.
    bundle_name = config.get("bundle_name")
    inject_depot: str | None = None
    inject_stops: list[str] | None = None
    if bundle_name:
        try:
            bundle = bundle_repo.read_bundle(graph_id, bundle_name)
        except NotFound as exc:
            raise ValueError(
                f"Bundle {bundle_name!r} not found for graph {graph_id!r}: {exc}"
            ) from exc
        inject_depot = bundle.get("depot")
        inject_stops = list(bundle.get("stops") or [])
        if not inject_depot or not inject_stops:
            raise ValueError(
                f"Bundle {bundle_name!r} is missing depot or stops; cannot inject."
            )

    # Stage 1 -- scenario generation
    job_service.mark_running(job_id, stage="scenario_gen")
    if bundle_name:
        job_service.emit_event(
            job_id,
            "scenario_gen",
            message=(
                f"generating {config['n_scenarios']} scenarios; "
                f"injecting bundle '{bundle_name}' as scenario 0"
            ),
        )
    else:
        job_service.emit_event(
            job_id, "scenario_gen", message=f"generating {config['n_scenarios']} scenarios"
        )

    await asyncio.to_thread(
        scenario_generator.generate_benchmark,
        graph_path=graph_path,
        graph_id=graph_id,
        config_path=config_path,
        benchmark_id=benchmark_id,
        out_dir=benchmark_dir,
        n_scenarios=int(config["n_scenarios"]),
        k_deliveries=int(config["k_deliveries"]),
        master_seed=int(config["master_seed"]),
        activation_strategy=config["activation_strategy"],
        sampler=config["sampler"],
        longitudinal=bool(config.get("longitudinal", False)),
        ri_keys=list(config["rain_intensities"]),
        inject_depot=inject_depot,
        inject_stops=inject_stops,
    )
    job_service.emit_event(job_id, "scenario_gen", message="scenarios written")

    # Stage 2 -- run policies
    job_repo.update_manifest(job_id, current_stage="run_policies")
    job_service.emit_event(
        job_id,
        "run_policies",
        total=len(algorithms),
        message=f"running {len(algorithms)} algorithm(s)",
    )

    await asyncio.to_thread(
        run_policies.run_policies,
        benchmark_dir=benchmark_dir,
        algorithm_ids=list(algorithms),
    )
    job_service.emit_event(
        job_id, "run_policies", message=f"completed {len(algorithms)} algorithm(s)"
    )

    # Stage 3 -- evaluate
    job_repo.update_manifest(job_id, current_stage="evaluate")
    job_service.emit_event(job_id, "evaluate", message="aggregating metrics")

    await asyncio.to_thread(
        evaluator.evaluate,
        benchmark_dir=benchmark_dir,
    )

    job_service.emit_event(job_id, "evaluate", message="done")
    job_service.mark_succeeded(
        job_id, result_path=f"benchmarks/{benchmark_id}"
    )


__all__ = ["run_benchmark_job", "DQNUnavailableError"]
