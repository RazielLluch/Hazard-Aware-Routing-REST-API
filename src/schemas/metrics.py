from pydantic import Field

from .common import AlgorithmId, CamelModel


class MetricBucket(CamelModel):
    """Per-(algorithm, RI, metric) summary statistics.

    Stat fields are nullable because the harness writes ``null`` when a
    bucket has zero observations -- e.g. ``hazard_exposure`` at RI4-RI5
    on benchmarks where every algorithm has 0% success. Coercing those
    nulls to 0.0 would conflate "no data" with "all values were 0", so
    the wire format keeps them as null and the frontend renders them as
    "--".
    """

    n: int
    mean: float | None = None
    stdev: float | None = None
    min: float | None = None
    max: float | None = None


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
