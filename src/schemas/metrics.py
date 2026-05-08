from pydantic import Field

from .common import AlgorithmId, CamelModel


class MetricBucket(CamelModel):
    n: int
    mean: float
    stdev: float = 0.0
    min: float = 0.0
    max: float = 0.0


class MetricsBundle(CamelModel):
    benchmark_id: str
    algorithms: dict[AlgorithmId, dict[str, dict[str, MetricBucket]]] = Field(default_factory=dict)


class LeaderboardEntry(CamelModel):
    algorithm_id: AlgorithmId
    rank: int
    metric_name: str
    metric_value: float
    success_rate: float
    failure_count: int
    n: int
