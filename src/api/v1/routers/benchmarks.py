from fastapi import APIRouter

from ....data import benchmark_repo
from ....schemas.benchmark import Benchmark, BenchmarkSummary

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[BenchmarkSummary])
def list_benchmarks() -> list[BenchmarkSummary]:
    return benchmark_repo.list_benchmark_summaries()


@router.get("/{benchmark_id}", response_model=Benchmark)
def get_benchmark(benchmark_id: str) -> Benchmark:
    return benchmark_repo.get_benchmark(benchmark_id)
