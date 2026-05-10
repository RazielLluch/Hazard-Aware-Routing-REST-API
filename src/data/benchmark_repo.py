import json
from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache
from typing import Any

from ..core import paths
from ..core.errors import NotFound
from ..schemas.benchmark import Benchmark, BenchmarkSummary, RIDistribution
from ..schemas.common import ri_from_int
from ..schemas.metrics import MetricBucket, MetricsBundle
from ..schemas.run import RouteEdge, Run, RunSummary
from ..schemas.scenario import Scenario, ScenarioListItem
from .jsonl import find_record, stream_jsonl


def list_benchmark_summaries() -> list[BenchmarkSummary]:
    out: list[BenchmarkSummary] = []
    for benchmark_id in paths.list_known_benchmark_ids():
        try:
            out.append(_load_benchmark(benchmark_id, summary_only=True))
        except FileNotFoundError:
            continue
    return out


def get_benchmark(benchmark_id: str) -> Benchmark:
    return _load_benchmark(benchmark_id, summary_only=False)  # type: ignore[return-value]


def _load_benchmark(benchmark_id: str, *, summary_only: bool) -> BenchmarkSummary | Benchmark:
    metadata_path = paths.benchmark_metadata_path(benchmark_id)
    if not metadata_path.exists():
        raise NotFound(f"Benchmark not found: {benchmark_id!r}")

    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    algorithms = _discover_algorithm_ids(benchmark_id)
    ri_dist = data.get("ri_distribution") or {}

    common_kwargs: dict[str, Any] = {
        "benchmark_id": data.get("benchmark_id", benchmark_id),
        "graph_id": data.get("graph_id", ""),
        "num_scenarios": data.get("n_scenarios", 0),
        "num_deliveries": data.get("k_deliveries", 0),
        "algorithms": algorithms,
        "ri_distribution": RIDistribution(
            RI1=ri_dist.get("RI1", 0),
            RI2=ri_dist.get("RI2", 0),
            RI3=ri_dist.get("RI3", 0),
            RI4=ri_dist.get("RI4", 0),
            RI5=ri_dist.get("RI5", 0),
        ),
    }

    if summary_only:
        return BenchmarkSummary(**common_kwargs)

    generated_at_raw = data.get("generated_at")
    generated_at = datetime.fromisoformat(generated_at_raw) if generated_at_raw else None

    return Benchmark(
        **common_kwargs,
        master_seed=data.get("master_seed", 0),
        sampling_policy=data.get("sampler", ""),
        activation_mode=data.get("activation_strategy", ""),
        feasibility_filtered=data.get("feasibility_filtered", False),
        graph_path=data.get("graph_path", ""),
        generated_at=generated_at,
    )


def _discover_algorithm_ids(benchmark_id: str) -> list[str]:
    runs_dir = paths.runs_dir(benchmark_id)
    if not runs_dir.exists():
        return []
    return sorted(
        p.stem for p in runs_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"
    )


def iter_scenarios(benchmark_id: str, *, ri: int | None = None) -> Iterator[Scenario]:
    path = paths.scenarios_path(benchmark_id)
    if not path.exists():
        raise NotFound(f"scenarios.jsonl not found for {benchmark_id!r}")
    for record in stream_jsonl(path):
        if ri is not None and record.get("rain_level") != ri:
            continue
        yield _scenario_from_dict(record, benchmark_id)


def get_scenario(benchmark_id: str, scenario_id: str) -> Scenario:
    path = paths.scenarios_path(benchmark_id)
    if not path.exists():
        raise NotFound(f"scenarios.jsonl not found for {benchmark_id!r}")
    record = find_record(path, lambda r: r.get("scenario_id") == scenario_id)
    if not record:
        raise NotFound(f"Scenario not found: {scenario_id!r}")
    return _scenario_from_dict(record, benchmark_id)


def list_scenario_summaries(
    benchmark_id: str,
    *,
    ri: int | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[ScenarioListItem], int]:
    path = paths.scenarios_path(benchmark_id)
    if not path.exists():
        raise NotFound(f"scenarios.jsonl not found for {benchmark_id!r}")

    items: list[ScenarioListItem] = []
    total = 0
    skip = max(0, (page - 1) * page_size)

    for record in stream_jsonl(path):
        if ri is not None and record.get("rain_level") != ri:
            continue
        total += 1
        if total <= skip:
            continue
        if len(items) < page_size:
            rl = int(record.get("rain_level", 1))
            items.append(
                ScenarioListItem(
                    scenario_id=record.get("scenario_id", ""),
                    rain_level=rl,
                    ri=ri_from_int(rl),
                    num_deliveries=len(record.get("delivery_nodes") or []),
                    num_blocked_edges=len(record.get("blocked_edges") or []),
                )
            )

    return items, total


def _scenario_from_dict(record: dict[str, Any], benchmark_id: str) -> Scenario:
    rl = int(record.get("rain_level", 1))
    return Scenario(
        scenario_id=record.get("scenario_id", ""),
        benchmark_id=benchmark_id,
        graph_id=record.get("graph_id", ""),
        rain_level=rl,
        ri=ri_from_int(rl),
        start_node=record.get("start_node", ""),
        delivery_nodes=record.get("delivery_nodes") or [],
        blocked_edges=[tuple(e) for e in (record.get("blocked_edges") or [])],
        max_steps=record.get("max_steps", 220),
        activation_mode=record.get("activation_mode", "deterministic_v3"),
        activation_seed=record.get("activation_seed"),
    )


def get_run(benchmark_id: str, scenario_id: str, algorithm_id: str) -> Run:
    path = paths.run_path(benchmark_id, algorithm_id)
    if not path.exists():
        raise NotFound(f"Run file not found: {algorithm_id!r}")
    record = find_record(path, lambda r: r.get("scenario_id") == scenario_id)
    if not record:
        raise NotFound(
            f"Run not found: scenario_id={scenario_id!r}, algorithm_id={algorithm_id!r}"
        )
    return _run_from_dict(record)


def list_run_summaries(benchmark_id: str, algorithm_id: str) -> list[RunSummary]:
    path = paths.run_path(benchmark_id, algorithm_id)
    if not path.exists():
        raise NotFound(f"Run file not found: {algorithm_id!r}")
    return [_run_summary_from_dict(record) for record in stream_jsonl(path)]


def list_runs_for_scenario(benchmark_id: str, scenario_id: str) -> list[RunSummary]:
    out: list[RunSummary] = []
    for algo_id in _discover_algorithm_ids(benchmark_id):
        path = paths.run_path(benchmark_id, algo_id)
        record = find_record(path, lambda r: r.get("scenario_id") == scenario_id)
        if record:
            out.append(_run_summary_from_dict(record))
    return out


def _run_summary_from_dict(record: dict[str, Any]) -> RunSummary:
    return RunSummary(
        scenario_id=record.get("scenario_id", ""),
        algorithm_id=record.get("algorithm_id", ""),  # type: ignore[arg-type]
        success=bool(record.get("success", False)),
        failure_reason=record.get("failure_reason"),
        replan_count=record.get("replan_count", 0),
        wall_time_ms=record.get("wall_time_ms", 0.0),
    )


def _run_from_dict(record: dict[str, Any]) -> Run:
    per_edge_raw = record.get("per_edge") or []
    per_edge = [
        RouteEdge(
            step=e.get("step", i),
            u=e.get("u", ""),
            v=e.get("v", ""),
            length_m=e.get("length_m", 0.0),
            travel_time=e.get("travel_time", 0.0),
            hazard_flood=e.get("hazard_flood", 0.0),
            hazard_landslide=e.get("hazard_landslide", 0.0),
            was_replan=bool(e.get("was_replan", False)),
        )
        for i, e in enumerate(per_edge_raw)
    ]
    return Run(
        scenario_id=record.get("scenario_id", ""),
        algorithm_id=record.get("algorithm_id", ""),  # type: ignore[arg-type]
        algorithm_config_hash=record.get("algorithm_config_hash"),
        success=bool(record.get("success", False)),
        failure_reason=record.get("failure_reason"),
        replan_count=record.get("replan_count", 0),
        wall_time_ms=record.get("wall_time_ms", 0.0),
        visit_order=record.get("visit_order") or [],
        edge_sequence=[tuple(e) for e in (record.get("edge_sequence") or [])],
        per_edge=per_edge,
        policy_metadata=record.get("policy_metadata") or {},
    )


@lru_cache(maxsize=8)
def get_metrics_bundle(benchmark_id: str) -> MetricsBundle:
    path = paths.metrics_json_path(benchmark_id)
    if not path.exists():
        raise NotFound(f"metrics.json not found for {benchmark_id!r}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    algos_raw = data.get("algorithms") or {}
    algorithms: dict[str, dict[str, dict[str, MetricBucket]]] = {}
    for algo_id, algo_data in algos_raw.items():
        metrics_raw = (algo_data or {}).get("metrics") or {}
        algorithms[algo_id] = {}
        for metric_name, ri_data in metrics_raw.items():
            algorithms[algo_id][metric_name] = {}
            for ri_key, bucket in (ri_data or {}).items():
                algorithms[algo_id][metric_name][ri_key] = MetricBucket(
                    n=bucket.get("n", 0),
                    mean=bucket.get("mean", 0.0),
                    stdev=bucket.get("stdev", 0.0),
                    min=bucket.get("min", 0.0),
                    max=bucket.get("max", 0.0),
                )

    return MetricsBundle(
        benchmark_id=benchmark_id,
        algorithms=algorithms,  # type: ignore[arg-type]
    )
