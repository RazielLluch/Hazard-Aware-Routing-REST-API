"""End-to-end smoke test for the vendored evaluation pipeline.

Runs in three layers, each callable independently or together:

  uv run python scripts/smoke_test_runners.py --layer all
  uv run python scripts/smoke_test_runners.py --layer direct
  uv run python scripts/smoke_test_runners.py --layer inference
  uv run python scripts/smoke_test_runners.py --layer jobs

Exits 0 on full pass; non-zero with structured failure report otherwise.

Layer descriptions:

- direct   : import each NNA runner, run it against a hand-rolled synthetic
             5-node scenario, assert Route validity per the contract. No
             uvicorn required. Adds Stage A coverage.
- inference: POST /api/v1/inference for every NNA x RI combination, assert
             coherent perEdge response. Requires uvicorn running. Adds
             Stage C coverage.
- jobs     : POST /api/v1/benchmarks for a 5-scenario cohort, poll
             /api/v1/jobs/{id} until done, validate filesystem + HTTP
             outputs. Requires uvicorn running. Adds Stage B coverage.

Layers `inference` and `jobs` are stubbed for now (the underlying API
endpoints land in Stage C and Stage B respectively); they print a
SKIPPED notice rather than failing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.runners.base import build_graph_view  # noqa: E402
from src.evaluation.runners.nna import NNADijkstra  # noqa: E402
from src.evaluation.runners.nna_astar import NNAAStar  # noqa: E402
from src.evaluation.runners.nna_astar_blind import NNAAStarBlind  # noqa: E402
from src.evaluation.runners.nna_blind import NNADijkstraBlind  # noqa: E402
from src.evaluation.runners.nna_ha import NNADijkstraHA  # noqa: E402
from src.evaluation.runners.nna_ha_blind import NNADijkstraHABlind  # noqa: E402
from src.evaluation.schemas import Route, Scenario, edge_key  # noqa: E402


FAILURE_REPORT_DIR = PROJECT_ROOT / "data" / "smoke_test_failures"

KNOWN_FAILURE_REASONS = {"trapped", "timeout", "invalid_action", "no_route", "blocked"}

# Hazard-blind plan-then-replan runners: plan on base_graph (which still
# contains blocked edges), so they encounter blocks at runtime and replan
# on the passable graph. These should record was_replan=True and
# planned_next_edge on the post-replan step.
NNA_REPLAN_ON_ENCOUNTER_RUNNERS = {
    "NNA-Dijkstra": NNADijkstra,
    "NNA-AStar": NNAAStar,
}

# Oracle runner: plans on view.activated_graph which already excludes
# blocked edges, so it never needs to replan (no replan_count, no
# was_replan markers). Should still produce a valid Route.
NNA_ORACLE_RUNNERS = {
    "NNA-Dijkstra-HA": NNADijkstraHA,
}

# Plan-then-fail-on-block runners: commit to a planned path on the full
# graph. First blocked-edge encounter ends the run with
# failure_reason="blocked". No replan, no recovery.
NNA_BLIND_RUNNERS = {
    "NNA-Dijkstra-Blind": NNADijkstraBlind,
    "NNA-AStar-Blind": NNAAStarBlind,
    "NNA-Dijkstra-HA-Blind": NNADijkstraHABlind,
}

ALL_RUNNERS = {**NNA_REPLAN_ON_ENCOUNTER_RUNNERS, **NNA_ORACLE_RUNNERS, **NNA_BLIND_RUNNERS}


@dataclass
class CaseFailure:
    layer: str
    case: str
    reason: str
    details: dict[str, Any]


def _build_synthetic_graph() -> nx.DiGraph:
    """A 5-node graph forming a small connected delivery network.

         A -- B -- C
         |    |
         D -- E

    Each edge gets length 1.0, base_time 1.0, and small hazard scores by
    default. Nodes have x,y coords for A* heuristic compatibility.
    """
    g = nx.DiGraph()
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0), "D": (0.0, -1.0), "E": (1.0, -1.0)}
    for node, (x, y) in coords.items():
        g.add_node(node, x=x, y=y)
    edges = [
        ("A", "B"), ("B", "A"),
        ("B", "C"), ("C", "B"),
        ("A", "D"), ("D", "A"),
        ("B", "E"), ("E", "B"),
        ("D", "E"), ("E", "D"),
    ]
    for u, v in edges:
        g.add_edge(
            u, v,
            length=1.0,
            base_time=1.0,
            travel_time=1.0,
            flood_hazard=0.1,
            landslide_hazard=0.05,
        )
    return g


def _make_replan_scenario(graph: nx.DiGraph) -> Scenario:
    """Force a replan: deliveries=[C], depot=A, edge (A,B) blocked.

    NNA plans A->B->C (the shortest path), hits a block on (A,B), and
    must replan via A->D->E->B->C. The first traversed step after the
    replan should record was_replan=True and planned_next_edge=(A,B).
    """
    travel_time_map = {edge_key(u, v): float(graph[u][v]["travel_time"]) for u, v in graph.edges()}
    blocked = [["A", "B"]]
    return Scenario(
        scenario_id="synthetic_replan",
        graph_id="synthetic",
        rain_level=2,
        activation_mode="deterministic_v3",
        activation_seed=1,
        start_node="A",
        delivery_nodes=["C"],
        blocked_edges=blocked,
        travel_time_map=travel_time_map,
        max_steps=20,
        num_deliveries=1,
        metadata={},
    )


def _make_trapped_scenario(graph: nx.DiGraph) -> Scenario:
    """Block every edge incident on the depot (A) so the agent cannot
    leave. Every runner should return success=False with a coherent
    failure_reason from KNOWN_FAILURE_REASONS.
    """
    travel_time_map = {edge_key(u, v): float(graph[u][v]["travel_time"]) for u, v in graph.edges()}
    blocked = [
        ["A", "B"], ["B", "A"],
        ["A", "D"], ["D", "A"],
    ]
    return Scenario(
        scenario_id="synthetic_trapped",
        graph_id="synthetic",
        rain_level=5,
        activation_mode="deterministic_v3",
        activation_seed=1,
        start_node="A",
        delivery_nodes=["C"],
        blocked_edges=blocked,
        travel_time_map=travel_time_map,
        max_steps=20,
        num_deliveries=1,
        metadata={},
    )


def _assert(cond: bool, msg: str, failures: list[CaseFailure], layer: str, case: str, **details: Any) -> None:
    if not cond:
        failures.append(CaseFailure(layer=layer, case=case, reason=msg, details=details))


def _validate_route(
    route: Route,
    expected_algorithm_id: str,
    delivery_nodes: list[str],
    failures: list[CaseFailure],
    case: str,
    *,
    layer: str = "direct",
) -> None:
    _assert(route is not None, "route is None", failures, layer, case)
    _assert(
        route.algorithm_id == expected_algorithm_id,
        f"algorithm_id mismatch: got {route.algorithm_id}, expected {expected_algorithm_id}",
        failures, layer, case,
    )
    _assert(isinstance(route.success, bool), "success not bool", failures, layer, case)
    if route.success:
        _assert(
            route.failure_reason is None,
            f"successful route has failure_reason={route.failure_reason}",
            failures, layer, case,
        )
        for d in delivery_nodes:
            _assert(
                d in route.visit_order,
                f"delivery {d} missing from visit_order",
                failures, layer, case, visit_order=route.visit_order,
            )
    else:
        _assert(
            route.failure_reason in KNOWN_FAILURE_REASONS,
            f"failure_reason {route.failure_reason!r} not in known set",
            failures, layer, case, known=sorted(KNOWN_FAILURE_REASONS),
        )

    _assert(route.wall_time_ms > 0.0, "wall_time_ms must be > 0", failures, layer, case)

    # Per-edge integrity (only when there are edges to check)
    for i, e in enumerate(route.per_edge):
        _assert(
            0.0 <= e["hazard_flood"] <= 1.0,
            f"per_edge[{i}].hazard_flood out of [0,1]: {e['hazard_flood']}",
            failures, layer, case,
        )
        _assert(
            0.0 <= e["hazard_landslide"] <= 1.0,
            f"per_edge[{i}].hazard_landslide out of [0,1]: {e['hazard_landslide']}",
            failures, layer, case,
        )
        _assert(e["travel_time"] > 0.0, f"per_edge[{i}].travel_time <= 0", failures, layer, case)
        _assert(e["length_m"] >= 0.0, f"per_edge[{i}].length_m < 0", failures, layer, case)
        if i == 0:
            _assert(e["step"] == 0, f"per_edge[0].step={e['step']} != 0", failures, layer, case)
        else:
            _assert(
                e["step"] == route.per_edge[i - 1]["step"] + 1,
                f"per_edge[{i}].step not monotonic",
                failures, layer, case,
            )


def run_layer_direct(failures: list[CaseFailure]) -> None:
    """Layer 1: direct in-process runner verification."""
    graph = _build_synthetic_graph()

    # Case A: replan-able scenario, every runner should produce a valid Route.
    scenario_replan = _make_replan_scenario(graph)
    view_replan = build_graph_view(graph, scenario_replan)

    for algo_id, cls in ALL_RUNNERS.items():
        case = f"replan-scenario / {algo_id}"
        try:
            policy = cls()
            route = policy.run(scenario_replan, view_replan)
        except Exception as exc:  # noqa: BLE001
            failures.append(CaseFailure(
                layer="direct", case=case, reason=f"runner raised: {exc}",
                details={"trace": traceback.format_exc()},
            ))
            continue

        _validate_route(
            route,
            expected_algorithm_id=algo_id,
            delivery_nodes=scenario_replan.delivery_nodes,
            failures=failures,
            case=case,
        )

        # Replan-on-encounter runners should succeed (passable detour exists)
        # and record at least one step with was_replan=True + planned_next_edge.
        if algo_id in NNA_REPLAN_ON_ENCOUNTER_RUNNERS:
            _assert(
                route.success,
                f"{algo_id} should succeed on synthetic replan scenario",
                failures, "direct", case, route=route.to_dict(),
            )
            _assert(
                route.replan_count > 0,
                f"{algo_id} expected replan_count>0 (depot edge A->B blocked)",
                failures, "direct", case,
            )
            replan_steps = [s for s in route.per_edge if s.get("was_replan")]
            _assert(
                len(replan_steps) >= 1,
                f"{algo_id} expected at least one was_replan=True step",
                failures, "direct", case, replan_count=route.replan_count,
            )
            if replan_steps:
                _assert(
                    replan_steps[0].get("planned_next_edge") is not None,
                    "first replan step missing planned_next_edge",
                    failures, "direct", case, step=replan_steps[0],
                )

        # Oracle runner plans on the passable graph (blocks excluded), so
        # it succeeds with replan_count=0 and no was_replan markers.
        if algo_id in NNA_ORACLE_RUNNERS:
            _assert(
                route.success,
                f"{algo_id} (oracle) should succeed on synthetic replan scenario",
                failures, "direct", case, route=route.to_dict(),
            )
            _assert(
                route.replan_count == 0,
                f"{algo_id} (oracle) should never replan; got replan_count={route.replan_count}",
                failures, "direct", case,
            )

        # Blind runners should fail with failure_reason="blocked" (they
        # plan A->B->C on the full graph and commit, hitting the blocked
        # A->B on step 0).
        if algo_id in NNA_BLIND_RUNNERS:
            _assert(
                not route.success,
                f"{algo_id} should fail (blocked) on synthetic replan scenario",
                failures, "direct", case, route=route.to_dict(),
            )
            _assert(
                route.failure_reason == "blocked",
                f"{algo_id} expected failure_reason='blocked', got {route.failure_reason!r}",
                failures, "direct", case,
            )

    # Case B: trapped scenario — every depot-incident edge blocked.
    scenario_trapped = _make_trapped_scenario(graph)
    view_trapped = build_graph_view(graph, scenario_trapped)

    for algo_id, cls in ALL_RUNNERS.items():
        case = f"trapped-scenario / {algo_id}"
        try:
            policy = cls()
            route = policy.run(scenario_trapped, view_trapped)
        except Exception as exc:  # noqa: BLE001
            failures.append(CaseFailure(
                layer="direct", case=case, reason=f"runner raised: {exc}",
                details={"trace": traceback.format_exc()},
            ))
            continue

        # Some runners may exit with empty per_edge; replan variants must
        # return success=False. Skip strict per-edge validation here since
        # per_edge can be empty.
        _assert(
            isinstance(route, Route),
            f"{algo_id} must return Route instance",
            failures, "direct", case,
        )
        _assert(
            not route.success,
            f"{algo_id} should fail on fully-blocked depot scenario",
            failures, "direct", case,
        )
        _assert(
            route.failure_reason in KNOWN_FAILURE_REASONS,
            f"{algo_id} unknown failure_reason {route.failure_reason!r}",
            failures, "direct", case,
        )


def run_layer_inference(failures: list[CaseFailure]) -> None:
    """Layer 2: HTTP /api/v1/inference verification. Stage C dependency."""
    print("[inference] SKIPPED (Stage C wires the live /api/v1/inference)")


def run_layer_jobs(failures: list[CaseFailure]) -> None:
    """Layer 3: HTTP /api/v1/benchmarks + /api/v1/jobs round-trip. Stage B dependency."""
    print("[jobs] SKIPPED (Stage B wires the in-process /jobs executor)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer",
        choices=["all", "direct", "inference", "jobs"],
        default="all",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args(argv)

    failures: list[CaseFailure] = []
    layers_to_run = ["direct", "inference", "jobs"] if args.layer == "all" else [args.layer]

    print(f"smoke_test_runners.py — running layer(s): {layers_to_run}")
    t0 = time.perf_counter()

    if "direct" in layers_to_run:
        print("\n=== Layer 1: direct runner verification ===")
        run_layer_direct(failures)

    if "inference" in layers_to_run:
        print("\n=== Layer 2: /api/v1/inference verification ===")
        run_layer_inference(failures)

    if "jobs" in layers_to_run:
        print("\n=== Layer 3: /api/v1/jobs round-trip verification ===")
        run_layer_jobs(failures)

    elapsed = time.perf_counter() - t0
    print(f"\n--- summary --- elapsed={elapsed:.2f}s")
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        for f in failures:
            print(f"  [{f.layer}] {f.case}: {f.reason}")
        FAILURE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        report_path = FAILURE_REPORT_DIR / f"{ts}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(
                [{"layer": x.layer, "case": x.case, "reason": x.reason, "details": x.details} for x in failures],
                f, indent=2, default=str,
            )
        print(f"report written: {report_path}")
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
