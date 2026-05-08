from pathlib import Path

from .config import settings


_KNOWN_GRAPHS = {
    "la_trinidad": "la_trinidad_hazard_graph.graphml",
    "la_trinidad_subgraph_n200": "staged_subgraphs/selected_subgraph_n200.graphml",
}


def _cohorts_root() -> Path:
    return settings.harness_root / "src" / "evaluation" / "cohorts"


def benchmark_dir(benchmark_id: str) -> Path:
    return _cohorts_root() / benchmark_id


def benchmark_metadata_path(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "cohort.json"


def scenarios_path(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "scenarios.jsonl"


def runs_dir(benchmark_id: str) -> Path:
    return benchmark_dir(benchmark_id) / "routes"


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
    return settings.graphs_root / _KNOWN_GRAPHS[graph_id]


def list_known_benchmark_ids() -> list[str]:
    cohorts_dir = _cohorts_root()
    if not cohorts_dir.exists():
        return []
    return sorted(
        p.name
        for p in cohorts_dir.iterdir()
        if p.is_dir() and (p / "cohort.json").exists()
    )


def list_known_graph_ids() -> list[str]:
    return sorted(_KNOWN_GRAPHS.keys())
