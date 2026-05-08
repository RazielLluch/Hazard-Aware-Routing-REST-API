from fastapi import APIRouter

from ....data import benchmark_repo
from ....schemas.metrics import MetricsBundle

router = APIRouter(prefix="/benchmarks/{benchmark_id}/metrics", tags=["metrics"])


@router.get("", response_model=MetricsBundle)
def get_metrics(benchmark_id: str) -> MetricsBundle:
    return benchmark_repo.get_metrics_bundle(benchmark_id)
