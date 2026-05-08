from fastapi import APIRouter

from ....data import benchmark_repo
from ....schemas.run import Run, RunSummary

router = APIRouter(prefix="/benchmarks/{benchmark_id}", tags=["runs"])


@router.get("/runs/{algorithm_id}", response_model=list[RunSummary])
def list_runs_for_algorithm(benchmark_id: str, algorithm_id: str) -> list[RunSummary]:
    return benchmark_repo.list_run_summaries(benchmark_id, algorithm_id)


@router.get("/scenarios/{scenario_id}/runs", response_model=list[RunSummary])
def list_runs_for_scenario(benchmark_id: str, scenario_id: str) -> list[RunSummary]:
    return benchmark_repo.list_runs_for_scenario(benchmark_id, scenario_id)


@router.get(
    "/scenarios/{scenario_id}/runs/{algorithm_id}",
    response_model=Run,
)
def get_run(benchmark_id: str, scenario_id: str, algorithm_id: str) -> Run:
    return benchmark_repo.get_run(benchmark_id, scenario_id, algorithm_id)
