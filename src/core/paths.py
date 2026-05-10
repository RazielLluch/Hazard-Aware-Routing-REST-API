from pathlib import Path

from .config import settings


_KNOWN_GRAPHS = {
    "la_trinidad": "la_trinidad_hazard_graph.graphml",
    "la_trinidad_subgraph_n200": "selected_subgraph_n200.graphml",
}


def project_root() -> Path:
    return settings.data_root.parent


def _benchmarks_root() -> Path:
    return settings.data_root / "benchmarks"


def benchmark_dir(benchmark_id: str) -> Path:
    return _benchmarks_root() / benchmark_id


def benchmark_metadata_path(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "benchmark.json"


def scenarios_path(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "scenarios.jsonl"


def runs_dir(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "runs"


def run_path(benchmark_id: str, algorithm_id: str) -> Path:
    return runs_dir(benchmark_id) / f"{algorithm_id}.jsonl"


def metrics_json_path(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "report" / "metrics.json"


def raw_metrics_csv_path(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "report" / "raw_metrics.csv"


def overall_metrics_csv_path(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "report" / "overall_metrics.csv"


def benchmark_cache_dir(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / ".cache"


def graph_path(graph_id: str) -> Path:
    if graph_id not in _KNOWN_GRAPHS:
        raise ValueError(f"Unknown graph_id: {graph_id!r}")
    return settings.data_root / "graphs" / _KNOWN_GRAPHS[graph_id]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def jobs_root() -> Path:
    return settings.data_root / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_root() / job_id


def job_manifest_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def job_events_path(job_id: str) -> Path:
    return job_dir(job_id) / "events.jsonl"


def list_known_job_ids() -> list[str]:
    root = jobs_root()
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "job.json").exists()
    )


# ---------------------------------------------------------------------------
# Node bundles (per-graph saved (depot, stops) tuples)
# ---------------------------------------------------------------------------


def bundles_root() -> Path:
    return settings.data_root / "node_bundles"


def bundle_dir(graph_id: str) -> Path:
    return bundles_root() / graph_id


def bundle_path(graph_id: str, name: str) -> Path:
    return bundle_dir(graph_id) / f"{name}.json"


def list_bundle_names(graph_id: str) -> list[str]:
    d = bundle_dir(graph_id)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.iterdir() if p.is_file() and p.suffix == ".json")


def list_known_benchmark_ids() -> list[str]:
    root = _benchmarks_root()
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "benchmark.json").exists()
    )


def list_known_graph_ids() -> list[str]:
    return sorted(_KNOWN_GRAPHS.keys())
