"""Pydantic schemas for benchmark-generation jobs.

Mirrors the on-disk shape at ``data/jobs/<job_id>/{job.json, events.jsonl}``.
Wire format is camelCase via ``CamelModel``; on-disk JSON uses snake_case
and ``populate_by_name`` lets the alias generator cover both.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from .common import AlgorithmId, CamelModel, RILevel


JobKind = Literal["benchmark_generation"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
JobStage = Literal["scenario_gen", "run_policies", "evaluate"]


class JobEvent(CamelModel):
    """One progress line appended to ``events.jsonl``."""

    ts: datetime
    stage: JobStage
    current: int = 0
    total: int = 0
    message: str = ""


class BenchmarkCreateRequest(CamelModel):
    """Request body for ``POST /api/v1/benchmarks``."""

    benchmark_id: str = Field(min_length=1, max_length=64)
    graph_id: str
    master_seed: int = 42
    n_scenarios: int = Field(default=100, ge=10, le=500)
    k_deliveries: int = Field(default=5, ge=2, le=10)
    rain_intensities: list[RILevel] = Field(
        default_factory=lambda: ["RI1", "RI2", "RI3", "RI4", "RI5"]
    )
    activation_strategy: str = "deterministic_v3"
    sampler: str = Field(
        default="uniform_open",
        pattern="^(uniform_open|scc_restricted)$",
    )
    longitudinal: bool = True
    algorithms: list[AlgorithmId] = Field(default_factory=list)
    # Optional: pre-fill scenario_idx 0 with a saved (depot, stops) bundle.
    # The bundle must already exist for the chosen graph_id (created via
    # /bundles or POST /api/v1/graphs/{graph_id}/bundles). When set, the
    # backend reads the bundle at job start and threads --inject-depot /
    # --inject-stops through to scenario_generator.
    bundle_name: str | None = Field(default=None, max_length=64)


class JobSummary(CamelModel):
    job_id: str
    kind: JobKind
    status: JobStatus
    started_at: datetime
    finished_at: datetime | None = None
    current_stage: JobStage | None = None
    result_path: str | None = None


class Job(JobSummary):
    config: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
