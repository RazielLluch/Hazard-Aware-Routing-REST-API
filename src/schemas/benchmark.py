from datetime import datetime

from pydantic import Field

from .common import AlgorithmId, CamelModel


class RIDistribution(CamelModel):
    RI1: int = 0
    RI2: int = 0
    RI3: int = 0
    RI4: int = 0
    RI5: int = 0


class BenchmarkSummary(CamelModel):
    benchmark_id: str
    graph_id: str
    num_scenarios: int
    num_deliveries: int
    algorithms: list[AlgorithmId] = Field(default_factory=list)
    ri_distribution: RIDistribution = Field(default_factory=RIDistribution)


class Benchmark(BenchmarkSummary):
    master_seed: int
    sampling_policy: str
    activation_mode: str
    feasibility_filtered: bool = True
    graph_path: str
    generated_at: datetime | None = None
