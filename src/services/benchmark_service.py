"""Benchmark generation orchestrator.

Spawns the three harness CLIs in sequence as subprocesses (clean isolation
per stage; no shared Python state with the FastAPI process) and emits
stage-level progress events to ``data/jobs/<job_id>/events.jsonl`` via
``job_service``. Stage-level granularity matches the SSE consumer's
three-stage progress UI; per-RI/per-algorithm events would need a
callback threaded through the harness.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..core.config import settings
from ..core.errors import NotFound
from ..data import bundle_repo, job_repo
from . import job_service


logger = logging.getLogger("services.benchmark")


# Map the request graph_id back to a graphml path on disk. Mirrors entries
# in core.paths._KNOWN_GRAPHS; we duplicate rather than import to keep the
# subprocess argv self-contained.
_GRAPH_PATHS = {
    "la_trinidad": "data/la_trinidad_hazard_graph.graphml",
    "la_trinidad_subgraph_n200": "data/staged_subgraphs/selected_subgraph_n200.graphml",
}

# Any one `_det` config works for scenario generation -- the rain_levels
# and time-weight fields are shared across profiles. We pick balanced as
# the canonical default.
_DEFAULT_CONFIG = (
    "src/evaluation/configs/hazard_training_final/balanced_HF/"
    "stage_200_balanced_HF_RI3_det.json"
)


def run_benchmark_job(job_id: str, config: dict[str, Any]) -> None:
    """Execute the full pipeline for one benchmark generation job.

    Designed to run in a FastAPI ``BackgroundTasks`` worker. Errors are
    captured and written to ``job.json`` so the SSE consumer can surface
    them; this function never raises.
    """
    try:
        _run_pipeline(job_id, config)
    except Exception as exc:  # noqa: BLE001 -- intentional catch-all at boundary
        logger.exception("benchmark job %s failed", job_id)
        job_service.emit_event(job_id, "evaluate", message=f"job failed: {exc}")
        job_service.mark_failed(job_id, error=str(exc))


def _run_pipeline(job_id: str, config: dict[str, Any]) -> None:
    benchmark_id = config["benchmark_id"]
    graph_id = config["graph_id"]
    if graph_id not in _GRAPH_PATHS:
        raise ValueError(f"Unknown graph_id: {graph_id!r}")

    harness_root = settings.harness_root
    benchmark_dir = harness_root / "src" / "evaluation" / "benchmarks" / benchmark_id

    # Optional: resolve a saved bundle into --inject-depot / --inject-stops
    # CLI flags. Validates up-front so a missing or wrong-graph bundle fails
    # the job before any scenario generation work happens.
    bundle_name = config.get("bundle_name")
    inject_args: list[str] = []
    if bundle_name:
        try:
            bundle = bundle_repo.read_bundle(graph_id, bundle_name)
        except NotFound as exc:
            raise ValueError(
                f"Bundle {bundle_name!r} not found for graph {graph_id!r}: {exc}"
            ) from exc
        depot = bundle.get("depot")
        stops = list(bundle.get("stops") or [])
        if not depot or not stops:
            raise ValueError(
                f"Bundle {bundle_name!r} is missing depot or stops; cannot inject."
            )
        inject_args = [
            "--inject-depot",
            depot,
            "--inject-stops",
            *stops,
        ]

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
    _run_subprocess(
        job_id,
        "scenario_gen",
        cwd=harness_root,
        argv=[
            sys.executable,
            "-m",
            "src.evaluation.scenario_generator",
            "--graph",
            _GRAPH_PATHS[graph_id],
            "--graph-id",
            graph_id,
            "--config",
            _DEFAULT_CONFIG,
            "--benchmark-id",
            benchmark_id,
            "--num-scenarios",
            str(config["n_scenarios"]),
            "--num-deliveries",
            str(config["k_deliveries"]),
            "--master-seed",
            str(config["master_seed"]),
            "--activation-strategy",
            config["activation_strategy"],
            "--sampler",
            config["sampler"],
            *(["--longitudinal"] if config.get("longitudinal") else []),
            "--ri-keys",
            *config["rain_intensities"],
            *inject_args,
        ],
    )

    # Stage 2 -- run policies
    algorithms = config.get("algorithms") or []
    if not algorithms:
        raise ValueError("at least one algorithm must be specified")

    job_repo.update_manifest(job_id, current_stage="run_policies")
    job_service.emit_event(
        job_id,
        "run_policies",
        total=len(algorithms),
        message=f"running {len(algorithms)} algorithm(s)",
    )
    _run_subprocess(
        job_id,
        "run_policies",
        cwd=harness_root,
        argv=[
            sys.executable,
            "-m",
            "src.evaluation.run_policies",
            "--benchmark-dir",
            str(benchmark_dir),
            "--algorithms",
            *algorithms,
        ],
    )

    # Stage 3 -- evaluate
    job_repo.update_manifest(job_id, current_stage="evaluate")
    job_service.emit_event(job_id, "evaluate", message="aggregating metrics")
    _run_subprocess(
        job_id,
        "evaluate",
        cwd=harness_root,
        argv=[
            sys.executable,
            "-m",
            "src.evaluation.evaluator",
            "--benchmark-dir",
            str(benchmark_dir),
        ],
    )

    job_service.emit_event(job_id, "evaluate", message="done")
    job_service.mark_succeeded(
        job_id, result_path=f"benchmarks/{benchmark_id}"
    )


def _run_subprocess(
    job_id: str,
    stage: str,
    *,
    cwd: Path,
    argv: list[str],
) -> None:
    """Run a harness CLI as a child process. Captures stdout/stderr and
    surfaces the last log line as the stage's final progress event. Raises
    CalledProcessError on non-zero exit so the outer handler can mark the
    job failed.
    """
    logger.info("job %s: %s -> %s", job_id, stage, " ".join(argv))
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-2000:]
        job_service.emit_event(
            job_id,
            stage,
            message=f"{stage} exited {completed.returncode}: {tail}",
        )
        raise subprocess.CalledProcessError(
            completed.returncode, argv, output=completed.stdout, stderr=completed.stderr
        )
    last_lines = (completed.stdout or "").strip().splitlines()
    summary = last_lines[-1] if last_lines else f"{stage} completed"
    job_service.emit_event(job_id, stage, message=summary)


__all__ = ["run_benchmark_job"]
