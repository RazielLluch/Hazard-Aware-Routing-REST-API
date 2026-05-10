from fastapi import APIRouter, BackgroundTasks, HTTPException

from ....data import benchmark_repo, job_repo
from ....evaluation import run_policies
from ....schemas.benchmark import Benchmark, BenchmarkSummary
from ....schemas.job import BenchmarkCreateRequest, JobSummary
from ....services.benchmark_service import run_benchmark_job

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[BenchmarkSummary])
def list_benchmarks() -> list[BenchmarkSummary]:
    return benchmark_repo.list_benchmark_summaries()


@router.get("/{benchmark_id}", response_model=Benchmark)
def get_benchmark(benchmark_id: str) -> Benchmark:
    return benchmark_repo.get_benchmark(benchmark_id)


@router.post("", response_model=JobSummary, status_code=202)
def create_benchmark(
    request: BenchmarkCreateRequest, background: BackgroundTasks
) -> JobSummary:
    """Kick off a benchmark generation job. Returns a JobSummary immediately
    (202 Accepted); the job runs in the background. Stream progress via
    ``GET /api/v1/jobs/{jobId}/stream``.

    Short-circuits with a 400 when DQN algorithms are requested but torch
    is not installed in this environment, so callers know the failure is
    deterministic rather than a runtime error inside the background job.
    """
    requested = list(request.algorithms or [])
    dqn_requested = [a for a in requested if a.startswith("DQN@")]
    if dqn_requested and not run_policies._HAS_TORCH:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "dqn_unavailable",
                "message": (
                    "DQN algorithms requested but torch is not installed in "
                    "this environment. Install torch (uv sync --extra dqn) "
                    "or remove DQN entries from the algorithms list."
                ),
                "requested": dqn_requested,
            },
        )

    job_id = job_repo.new_job_id(prefix="bench")
    config = request.model_dump(by_alias=False)
    manifest = job_repo.create_job(
        job_id=job_id, kind="benchmark_generation", config=config
    )
    background.add_task(run_benchmark_job, job_id, config)
    return JobSummary(**{
        k: manifest.get(k) for k in (
            "job_id", "kind", "status", "started_at", "finished_at",
            "current_stage", "result_path",
        )
    })
