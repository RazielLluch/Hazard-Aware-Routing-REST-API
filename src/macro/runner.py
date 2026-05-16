# ============================================================================
# VENDORED FILE — do not edit by hand.
# Verbatim copy of macro_DDQN/run_macro_dqn_final.py, vendored 2026-05-15.
# The ONLY changes from the canonical source are this header and the marked
# "VENDOR PATCH" block below (REPO_ROOT made configurable). Re-sync by
# re-copying the canonical file and re-applying the patch.
# Drift guard: tests/macro/test_runner_fidelity.py.
# ============================================================================
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# === VENDOR PATCH (web/Hazard-Aware-Routing-REST-API/src/macro/) ===
# Canonical line: REPO_ROOT = Path(__file__).resolve().parents[1]
# Vendored under src/macro/, REPO_ROOT points at the standalone vendor tree at
# ``data/macro_vendor/`` which mirrors the canonical layout (``macro_DDQN/`` +
# ``map_preprocessing/``), so ``resolve_path()`` finds the hazard graph and
# artefacts without any config mutations. ``bridge.py`` sets ``MACRO_DDQN_ROOT``
# from ``settings.macro_vendor_root``; the fallback maps
# ``.../src/macro/runner.py`` -> ``.../data/macro_vendor`` for direct CLI use.
REPO_ROOT = Path(
    os.environ.get("MACRO_DDQN_ROOT")
    or Path(__file__).resolve().parents[2] / "data" / "macro_vendor"
).resolve()
# === END VENDOR PATCH ===

FEATURE_NAMES = [
    "current_x",
    "current_y",
    "target_x",
    "target_y",
    "delta_x",
    "delta_y",
    "euclidean_xy",
    "remaining_frac",
    "rain_norm",
    "eta_time",
    "hazard_lambda_norm",
    "w_flood",
    "w_landslide",
    "path_objective_log",
    "path_time_log",
    "path_distance_km_log",
    "path_flood_avg",
    "path_landslide_avg",
    "path_risk_avg",
    "path_hops_frac",
    "future_min_objective_log",
    "future_mean_objective_log",
    "future_min_time_log",
    "future_mean_time_log",
    "future_mean_risk",
    "future_greedy_objective_log",
    "future_greedy_time_log",
    "future_greedy_risk_avg",
    "future_greedy_hops_frac",
    "candidate_greedy_completion_log",
    "current_objective_rank",
    "current_time_rank",
]


@dataclass(frozen=True)
class RainLevel:
    key: str
    speed_mult: float
    flood_block_threshold: float
    flood_block_prob: float
    landslide_block_threshold: float
    landslide_block_prob: float

    @classmethod
    def from_config(cls, key: str, cfg: dict[str, Any]) -> "RainLevel":
        return cls(
            key=key,
            speed_mult=float(cfg["speed_mult"]),
            flood_block_threshold=float(cfg["flood_block_threshold"]),
            flood_block_prob=float(cfg["flood_block_prob"]),
            landslide_block_threshold=float(cfg["landslide_block_threshold"]),
            landslide_block_prob=float(cfg["landslide_block_prob"]),
        )


@dataclass(frozen=True)
class Profile:
    key: str
    eta_time: float
    hazard_lambda: float
    w_flood: float
    w_landslide: float
    step_cost: float
    lambda_soft: float | None = None
    lambda_blockage: float | None = None

    @classmethod
    def from_config(cls, key: str, cfg: dict[str, Any]) -> "Profile":
        return cls(
            key=key,
            eta_time=float(cfg["eta_time"]),
            hazard_lambda=float(cfg["hazard_lambda"]),
            w_flood=float(cfg["w_flood"]),
            w_landslide=float(cfg["w_landslide"]),
            step_cost=float(cfg.get("step_cost", 0.0)),
            lambda_soft=(
                float(cfg["lambda_soft"])
                if cfg.get("lambda_soft") is not None
                else None
            ),
            lambda_blockage=(
                float(cfg["lambda_blockage"])
                if cfg.get("lambda_blockage") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    ri_key: str
    start: str
    deliveries: tuple[str, ...]
    max_steps: int


@dataclass(frozen=True)
class PathMetrics:
    path: tuple[str, ...]
    objective: float
    travel_time_min: float
    distance_m: float
    flood_exposure: float
    landslide_exposure: float
    blockage_exposure: float
    risk_exposure: float
    hops: int

    @staticmethod
    def empty(start: str) -> "PathMetrics":
        return PathMetrics((start,), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


@dataclass
class RouteResult:
    scenario_id: str
    ri_key: str
    profile: str
    algorithm: str
    success: bool
    failure_reason: str
    delivered: int
    objective: float
    travel_time_min: float
    distance_m: float
    flood_exposure: float
    landslide_exposure: float
    blockage_exposure: float
    risk_exposure: float
    common_risk_exposure: float
    hops: int
    replan_count: int
    visit_order: str
    wall_time_ms: float

    def to_row(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "ri_key": self.ri_key,
            "profile": self.profile,
            "algorithm": self.algorithm,
            "success": int(self.success),
            "failure_reason": self.failure_reason,
            "delivered": self.delivered,
            "objective": round(self.objective, 6),
            "travel_time_min": round(self.travel_time_min, 6),
            "distance_m": round(self.distance_m, 6),
            "flood_exposure": round(self.flood_exposure, 6),
            "landslide_exposure": round(self.landslide_exposure, 6),
            "blockage_exposure": round(self.blockage_exposure, 6),
            "risk_exposure": round(self.risk_exposure, 6),
            "common_risk_exposure": round(self.common_risk_exposure, 6),
            "hops": self.hops,
            "replan_count": self.replan_count,
            "visit_order": self.visit_order,
            "wall_time_ms": round(self.wall_time_ms, 6),
        }


class CostToGoNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(int(width)))
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GraphBundle:
    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph
        xs = [float(data.get("x", 0.0)) for _, data in graph.nodes(data=True)]
        ys = [float(data.get("y", 0.0)) for _, data in graph.nodes(data=True)]
        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)
        times = [float(data["base_time"]) for _, _, data in graph.edges(data=True)]
        self.mean_edge_time = statistics.fmean(times) if times else 1.0
        self.max_hops_scale = max(10.0, float(graph.number_of_nodes()) / 4.0)

    def xy_norm(self, node: str) -> tuple[float, float]:
        data = self.graph.nodes[node]
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        nx_ = (x - self.min_x) / max(self.max_x - self.min_x, 1e-9)
        ny_ = (y - self.min_y) / max(self.max_y - self.min_y, 1e-9)
        return nx_, ny_


class ActivatedNetwork:
    def __init__(
        self,
        bundle: GraphBundle,
        rain: RainLevel,
        profile: Profile,
        alpha_flood: float,
        alpha_landslide: float,
        risk_model: dict[str, Any],
        hazard_cfg: dict[str, Any],
        activation_seed: int,
    ) -> None:
        self.bundle = bundle
        self.rain = rain
        self.profile = profile
        self.alpha_flood = alpha_flood
        self.alpha_landslide = alpha_landslide
        self.risk_model = risk_model
        self.hazard_cfg = hazard_cfg
        self.activation_seed = activation_seed
        self.objective_mode = str(risk_model.get("objective_mode", "combined"))
        self.activation_model = str(
            hazard_cfg.get("activation_model", hazard_cfg.get("activation_mode", "legacy"))
        )
        self.uses_threshold_table_v3 = self.activation_model == "threshold_table_v3"
        self.activation_thresholds = hazard_cfg.get("activation_thresholds", {})
        soft_blocking = hazard_cfg.get("soft_blocking", {})
        self.soft_blocking_enabled = bool(soft_blocking.get("enabled", False))
        self.plan_with_soft_blocks = bool(soft_blocking.get("plan_with_soft_blocks", False))
        self.blockage_risk_multiplier = float(soft_blocking.get("blockage_risk_multiplier", 50.0))
        self.blocked_time_multiplier = float(soft_blocking.get("blocked_time_multiplier", 8.0))
        hard_blocking = hazard_cfg.get("hard_blocking", {})
        self.hard_blocking_enabled = bool(hard_blocking.get("enabled", False))
        self.fail_on_blocked_traversal = bool(
            hard_blocking.get("fail_on_blocked_traversal", self.hard_blocking_enabled)
        )
        if self.hard_blocking_enabled:
            self.plan_with_soft_blocks = False
        self.common_hazard_weight = float(risk_model.get("common_hazard_weight", 0.5))
        self.stochastic_cfg = hazard_cfg.get("stochastic_evaluation", {})
        self.stochastic_enabled = bool(self.stochastic_cfg.get("enabled", False))
        self.full_graph = nx.DiGraph()
        self.passable_graph = nx.DiGraph()
        self.blocked_edges: set[tuple[str, str]] = set()
        self._path_cache: dict[tuple[str, str, str, bool, str], PathMetrics | None] = {}
        self._build_graphs()

    def _build_graphs(self) -> None:
        base = self.bundle.graph
        self.full_graph.add_nodes_from(base.nodes(data=True))
        self.passable_graph.add_nodes_from(base.nodes(data=True))

        blocked_pairs = set() if self.uses_threshold_table_v3 else self._blocked_canonical_pairs()
        for u, v, data in base.edges(data=True):
            new_data = dict(data)
            hf = float(new_data.get("flood_hazard", 0.0))
            hl = float(new_data.get("landslide_hazard", 0.0))
            flood_active, landslide_active, flood_blocked, landslide_blocked = self._activation_status(
                str(u), str(v), hf, hl
            )
            blocked = (
                canonical_edge(u, v) in blocked_pairs
                or flood_blocked
                or landslide_blocked
            )
            active_hf = hf if flood_active else 0.0
            active_hl = hl if landslide_active else 0.0
            travel_time = (
                float(new_data["base_time"])
                / max(self.rain.speed_mult, 1e-9)
                * (1.0 + self.alpha_flood * active_hf + self.alpha_landslide * active_hl)
            )
            if blocked and (self.soft_blocking_enabled or self.uses_threshold_table_v3):
                travel_time *= self.blocked_time_multiplier
            flood_exposure, landslide_exposure, blockage_exposure, risk = self._edge_risk_components(
                hf=hf,
                hl=hl,
                length_m=float(new_data["length_m"]),
                flood_active=flood_active,
                landslide_active=landslide_active,
                blocked=blocked,
            )
            soft_hazard_exposure = (
                self.profile.w_flood * flood_exposure
                + self.profile.w_landslide * landslide_exposure
            )
            time_cost = self.profile.eta_time * (
                travel_time / max(self.bundle.mean_edge_time, 1e-9)
            )
            if (
                self.objective_mode == "separated"
                or self.profile.lambda_soft is not None
                or self.profile.lambda_blockage is not None
            ):
                lambda_soft = (
                    self.profile.lambda_soft
                    if self.profile.lambda_soft is not None
                    else self.profile.hazard_lambda
                )
                lambda_blockage = (
                    self.profile.lambda_blockage
                    if self.profile.lambda_blockage is not None
                    else self.profile.hazard_lambda
                )
                objective = (
                    time_cost
                    + lambda_soft * soft_hazard_exposure
                    + lambda_blockage * blockage_exposure
                    + self.profile.step_cost
                )
            else:
                objective = time_cost + self.profile.hazard_lambda * risk + self.profile.step_cost
            new_data["travel_time"] = travel_time
            new_data["risk"] = risk
            new_data["soft_hazard_exposure"] = soft_hazard_exposure
            new_data["flood_exposure"] = flood_exposure
            new_data["landslide_exposure"] = landslide_exposure
            new_data["blockage_exposure"] = blockage_exposure
            new_data["flood_active"] = int(flood_active)
            new_data["landslide_active"] = int(landslide_active)
            new_data["flood_blocked"] = int(flood_blocked)
            new_data["landslide_blocked"] = int(landslide_blocked)
            new_data["blocked"] = int(blocked)
            new_data["objective"] = objective
            self.full_graph.add_edge(u, v, **new_data)

            if blocked:
                self.blocked_edges.add((u, v))
                continue
            self.passable_graph.add_edge(u, v, **new_data)

    def _activation_status(
        self,
        u: str,
        v: str,
        hf: float,
        hl: float,
    ) -> tuple[bool, bool, bool, bool]:
        if not self.uses_threshold_table_v3:
            return True, True, False, False

        ri = int(self.rain.key.replace("RI", ""))

        def ge(value: float, threshold: float) -> bool:
            return value >= threshold - 1e-9

        flood_active_t = self._threshold_for(
            "flood_active",
            ri,
            {3: 1.0, 4: 0.6, 5: 0.2},
        )
        flood_block_t = self._threshold_for(
            "flood_block",
            ri,
            {4: 1.0, 5: 0.6},
        )
        landslide_active_t = self._threshold_for(
            "landslide_active",
            ri,
            {2: 1.0, 3: 0.6, 4: 0.2, 5: 0.2},
        )
        landslide_block_t = self._threshold_for(
            "landslide_block",
            ri,
            {2: 1.0, 3: 0.6, 4: 0.2, 5: 0.2},
        )

        flood_active = flood_active_t is not None and ge(hf, flood_active_t)
        flood_blocked = (
            flood_active
            and flood_block_t is not None
            and ge(hf, flood_block_t)
        )
        landslide_active = landslide_active_t is not None and ge(hl, landslide_active_t)
        landslide_blocked = (
            landslide_active
            and landslide_block_t is not None
            and ge(hl, landslide_block_t)
        )
        if self.stochastic_enabled:
            flood_active, landslide_active, flood_blocked, landslide_blocked = (
                self._stochastic_activation_status(
                    u=u,
                    v=v,
                    hf=hf,
                    hl=hl,
                    flood_active=flood_active,
                    landslide_active=landslide_active,
                    flood_blocked=flood_blocked,
                    landslide_blocked=landslide_blocked,
                )
            )
        return flood_active, landslide_active, flood_blocked, landslide_blocked

    def _threshold_for(
        self,
        table_name: str,
        ri: int,
        defaults: dict[int, float],
    ) -> float | None:
        table = self.activation_thresholds.get(table_name, {})
        keys = (f"RI{ri}", str(ri), ri)
        for key in keys:
            if key in table:
                value = table[key]
                return None if value is None else float(value)
        return defaults.get(ri)

    def _stochastic_activation_status(
        self,
        u: str,
        v: str,
        hf: float,
        hl: float,
        flood_active: bool,
        landslide_active: bool,
        flood_blocked: bool,
        landslide_blocked: bool,
    ) -> tuple[bool, bool, bool, bool]:
        mode = str(self.stochastic_cfg.get("mode", "activation_and_blockage"))
        stochastic_activation = mode in {"activation", "activation_and_blockage", "both"}
        stochastic_blockage = mode in {"blockage", "activation_and_blockage", "both"}

        if stochastic_activation:
            if flood_active:
                flood_active = self._edge_random(u, v, "flood_active") < self._stochastic_probability(
                    "activation_probability", "flood", hf, 1.0
                )
            if landslide_active:
                landslide_active = self._edge_random(u, v, "landslide_active") < self._stochastic_probability(
                    "activation_probability", "landslide", hl, 1.0
                )
        flood_blocked = flood_active and flood_blocked
        landslide_blocked = landslide_active and landslide_blocked

        if stochastic_blockage:
            if flood_blocked:
                flood_blocked = self._edge_random(u, v, "flood_blocked") < self._stochastic_probability(
                    "block_probability", "flood", hf, 1.0
                )
            if landslide_blocked:
                landslide_blocked = self._edge_random(u, v, "landslide_blocked") < self._stochastic_probability(
                    "block_probability", "landslide", hl, 1.0
                )
        return flood_active, landslide_active, flood_blocked, landslide_blocked

    def _edge_random(self, u: str, v: str, salt: str) -> float:
        a, b = canonical_edge(u, v)
        key = f"{self.activation_seed}|{self.rain.key}|{a}|{b}|{salt}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) / float(16 ** 16)

    def _stochastic_probability(
        self,
        table_name: str,
        hazard_kind: str,
        hazard_score: float,
        default: float,
    ) -> float:
        table = self.stochastic_cfg.get(table_name, {})
        kind_table = table.get(hazard_kind, {})
        ri_table = kind_table.get(self.rain.key, {})
        keys = [f"{hazard_score:.1f}", str(hazard_score)]
        value = None
        for key in keys:
            if key in ri_table:
                value = ri_table[key]
                break
        if value is None:
            value = kind_table.get("default", table.get("default", default))
        return min(1.0, max(0.0, float(value)))

    def _edge_risk_components(
        self,
        hf: float,
        hl: float,
        length_m: float,
        flood_active: bool,
        landslide_active: bool,
        blocked: bool,
    ) -> tuple[float, float, float, float]:
        flood_power = float(self.risk_model.get("flood_exponent", 1.0))
        landslide_power = float(self.risk_model.get("landslide_exponent", 1.0))
        distance_weighted = bool(self.risk_model.get("distance_weighted", False))
        distance_unit_m = float(self.risk_model.get("distance_unit_m", 1000.0))
        distance_factor = length_m / max(distance_unit_m, 1e-9) if distance_weighted else 1.0
        ri_multipliers = self.risk_model.get("ri_risk_multipliers", {})
        ri_multiplier = float(ri_multipliers.get(self.rain.key, 1.0))
        active_hf = max(hf, 0.0) if flood_active else 0.0
        active_hl = max(hl, 0.0) if landslide_active else 0.0
        flood_value = self._hazard_metric_value(active_hf, flood_power)
        landslide_value = self._hazard_metric_value(active_hl, landslide_power)
        flood_exposure = ri_multiplier * distance_factor * flood_value
        landslide_exposure = ri_multiplier * distance_factor * landslide_value
        blockage_exposure = 0.0
        if blocked and (self.soft_blocking_enabled or self.uses_threshold_table_v3):
            blockage_exposure = self.blockage_risk_multiplier * distance_factor
        risk = (
            self.profile.w_flood * flood_exposure
            + self.profile.w_landslide * landslide_exposure
            + blockage_exposure
        )
        return flood_exposure, landslide_exposure, blockage_exposure, risk

    def _hazard_metric_value(self, hazard_score: float, exponent: float) -> float:
        mode = str(self.risk_model.get("hazard_value_mode", "power"))
        if mode != "category_weights":
            return max(hazard_score, 0.0) ** exponent

        weights = self.risk_model.get("category_weights", {})
        key = f"{hazard_score:.1f}"
        if key in weights:
            return float(weights[key])
        for source, target in weights.items():
            try:
                if abs(hazard_score - float(source)) <= 1e-9:
                    return float(target)
            except (TypeError, ValueError):
                continue
        return max(hazard_score, 0.0) ** exponent

    def _blocked_canonical_pairs(self) -> set[tuple[str, str]]:
        rng = random.Random(self.activation_seed)
        pair_hazards: dict[tuple[str, str], tuple[float, float]] = {}
        for u, v, data in self.bundle.graph.edges(data=True):
            key = canonical_edge(u, v)
            pair_hazards[key] = (
                max(pair_hazards.get(key, (0.0, 0.0))[0], float(data.get("flood_hazard", 0.0))),
                max(pair_hazards.get(key, (0.0, 0.0))[1], float(data.get("landslide_hazard", 0.0))),
            )

        blocked: set[tuple[str, str]] = set()
        for key in sorted(pair_hazards):
            hf, hl = pair_hazards[key]
            flood_block = False
            if hf >= self.rain.flood_block_threshold:
                flood_block = rng.random() < self.rain.flood_block_prob
            landslide_block = False
            if hl >= self.rain.landslide_block_threshold:
                landslide_block = rng.random() < self.rain.landslide_block_prob
            if flood_block or landslide_block:
                blocked.add(key)
        return blocked

    def path(
        self,
        source: str,
        target: str,
        weight: str = "objective",
        passable: bool = True,
    ) -> PathMetrics | None:
        graph, graph_key = self._graph_for_path(weight, passable)
        key = (source, target, weight, passable, graph_key)
        if key in self._path_cache:
            return self._path_cache[key]
        try:
            nodes = nx.dijkstra_path(graph, source, target, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self._path_cache[key] = None
            return None
        metrics = self.metrics_for_path(nodes, passable=passable, weight=weight)
        self._path_cache[key] = metrics
        return metrics

    def _graph_for_path(self, weight: str, passable: bool) -> tuple[nx.DiGraph, str]:
        if passable and self.plan_with_soft_blocks:
            return self.full_graph, "soft_full"
        if passable:
            return self.passable_graph, "passable"
        return self.full_graph, "full"

    def metrics_for_path(
        self,
        nodes: Iterable[str],
        passable: bool = True,
        weight: str = "objective",
    ) -> PathMetrics:
        path = tuple(nodes)
        if not path:
            return PathMetrics((), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        if len(path) <= 1:
            return PathMetrics.empty(path[0])
        graph, _ = self._graph_for_path(weight, passable)
        objective = 0.0
        travel_time = 0.0
        distance = 0.0
        flood = 0.0
        landslide = 0.0
        blockage = 0.0
        risk = 0.0
        for u, v in zip(path, path[1:]):
            data = graph[u][v]
            objective += float(data["objective"])
            travel_time += float(data["travel_time"])
            distance += float(data["length_m"])
            flood += float(data.get("flood_exposure", data["flood_hazard"]))
            landslide += float(data.get("landslide_exposure", data["landslide_hazard"]))
            blockage += float(data.get("blockage_exposure", 0.0))
            risk += float(data["risk"])
        return PathMetrics(
            path=path,
            objective=objective,
            travel_time_min=travel_time,
            distance_m=distance,
            flood_exposure=flood,
            landslide_exposure=landslide,
            blockage_exposure=blockage,
            risk_exposure=risk,
            hops=len(path) - 1,
        )

    def planned_path_hits_blocked_edge(self, nodes: Iterable[str]) -> bool:
        path = tuple(nodes)
        for u, v in zip(path, path[1:]):
            if not self.full_graph.has_edge(u, v):
                return True
            if int(self.full_graph[u][v].get("blocked", 0)) == 1:
                return True
        return False

    def execute_planned_path(
        self,
        nodes: Iterable[str],
        passable: bool = False,
        weight: str = "objective",
    ) -> PathMetrics | None:
        path = tuple(nodes)
        if self.fail_on_blocked_traversal and self.planned_path_hits_blocked_edge(path):
            return None
        return self.metrics_for_path(path, passable=passable, weight=weight)


def canonical_edge(u: str, v: str) -> tuple[str, str]:
    return (u, v) if str(u) <= str(v) else (v, u)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_seed_offset(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def float_attr(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def load_hazard_graph(cfg: dict[str, Any]) -> GraphBundle:
    graph_cfg = cfg["graph"]
    normalization_cfg = cfg.get("hazard", {}).get("normalization", {})
    graph_path = resolve_path(graph_cfg["path"])
    raw = nx.read_graphml(graph_path)
    graph = nx.DiGraph()
    for node, data in raw.nodes(data=True):
        graph.add_node(
            str(node),
            x=float_attr(data, "x"),
            y=float_attr(data, "y"),
            street_count=float_attr(data, "street_count"),
        )

    flood_attr = graph_cfg.get("flood_attr", "flood_hazard")
    landslide_attr = graph_cfg.get("landslide_attr", "landslide_hazard")
    time_attr = graph_cfg.get("travel_time_attr", "travel_time_min")
    length_attr = graph_cfg.get("length_attr", "length")

    def normalized_edge_data(data: dict[str, Any]) -> dict[str, Any]:
        length_m = float_attr(data, length_attr)
        base_time = float_attr(data, time_attr)
        if base_time <= 0.0:
            # Fallback to 30 kph when an OSM-derived travel time is missing.
            base_time = (length_m / 1000.0) / 30.0 * 60.0
        hf = normalize_hazard_value(
            float_attr(data, flood_attr),
            normalization_cfg.get("flood_map", {}),
        )
        hl = normalize_hazard_value(
            float_attr(data, landslide_attr),
            normalization_cfg.get("landslide_map", {}),
        )
        return {
            "length_m": length_m,
            "base_time": base_time,
            "flood_hazard": hf,
            "landslide_hazard": hl,
            "combined_hazard": float_attr(data, "combined_hazard", 0.6 * hf + 0.4 * hl),
        }

    if raw.is_directed():
        for u, v, data in raw.edges(data=True):
            graph.add_edge(str(u), str(v), **normalized_edge_data(data))
    else:
        for u, v, data in raw.edges(data=True):
            edge_data = normalized_edge_data(data)
            graph.add_edge(str(u), str(v), **edge_data)
            graph.add_edge(str(v), str(u), **edge_data)

    return GraphBundle(graph)


def normalize_hazard_value(value: float, mapping: dict[str, Any]) -> float:
    for source, target in mapping.items():
        try:
            source_value = float(source)
            target_value = float(target)
        except (TypeError, ValueError):
            continue
        if abs(value - source_value) <= 1e-9:
            return target_value
    return value


def parse_rain_levels(cfg: dict[str, Any]) -> dict[str, RainLevel]:
    return {
        key: RainLevel.from_config(key, value)
        for key, value in cfg["hazard"]["rain_levels"].items()
    }


def parse_profiles(cfg: dict[str, Any]) -> dict[str, Profile]:
    return {
        key: Profile.from_config(key, value)
        for key, value in cfg["profiles"].items()
    }


def make_networks(
    bundle: GraphBundle,
    rains: dict[str, RainLevel],
    profiles: dict[str, Profile],
    cfg: dict[str, Any],
    ri_keys: list[str],
    profile_keys: list[str],
    seed_offset: int = 0,
) -> dict[tuple[str, str], ActivatedNetwork]:
    drag = cfg["hazard"].get("travel_time_drag", {})
    risk_model = cfg["hazard"].get("risk_model", {})
    alpha_flood = float(drag.get("alpha_flood", 0.5))
    alpha_landslide = float(drag.get("alpha_landslide", 0.5))
    base_seed = int(cfg.get("seed", 0))
    networks: dict[tuple[str, str], ActivatedNetwork] = {}
    for ri_key in ri_keys:
        ri_number = int(ri_key.replace("RI", ""))
        for profile_key in profile_keys:
            networks[(ri_key, profile_key)] = ActivatedNetwork(
                bundle=bundle,
                rain=rains[ri_key],
                profile=profiles[profile_key],
                alpha_flood=alpha_flood,
                alpha_landslide=alpha_landslide,
                risk_model=risk_model,
                hazard_cfg=cfg["hazard"],
                activation_seed=base_seed + seed_offset + ri_number,
            )
    return networks


def generate_scenarios(
    networks: dict[tuple[str, str], ActivatedNetwork],
    ri_keys: list[str],
    profile_for_sampling: str,
    scenarios_per_ri: int,
    num_deliveries: int,
    max_steps: int,
    seed: int,
    scenario_prefix: str,
    sampling_mode: str = "largest_passable_component",
    sampling_options: dict[str, Any] | None = None,
) -> list[Scenario]:
    rng = random.Random(seed)
    sampling_options = sampling_options or {}
    scenarios: list[Scenario] = []
    for ri_key in ri_keys:
        network = networks[(ri_key, profile_for_sampling)]
        graph = (
            network.full_graph
            if sampling_mode in {
                "full_graph",
                "full_graph_largest_component",
                "hazard_opportunity",
                "hazard_opportunity_full_graph",
            }
            else network.passable_graph
        )
        components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
        if not components:
            raise RuntimeError(f"{ri_key} has no component for sampling mode '{sampling_mode}'.")
        pool = sorted(components[0])
        if len(pool) < num_deliveries + 1:
            raise RuntimeError(
                f"{ri_key} sampling pool '{sampling_mode}' has {len(pool)} nodes, "
                f"but {num_deliveries + 1} are needed."
            )
        if sampling_mode in {
            "hazard_opportunity",
            "hazard_opportunity_full_graph",
            "hazard_opportunity_passable",
        }:
            accepted = 0
            attempts = 0
            max_attempts = int(
                sampling_options.get(
                    "max_attempts",
                    scenarios_per_ri * int(sampling_options.get("attempt_multiplier", 120)),
                )
            )
            while accepted < scenarios_per_ri and attempts < max_attempts:
                attempts += 1
                sample = rng.sample(pool, num_deliveries + 1)
                candidate = Scenario(
                    scenario_id=f"{scenario_prefix}_{ri_key}_{accepted:05d}",
                    ri_key=ri_key,
                    start=sample[0],
                    deliveries=tuple(sample[1:]),
                    max_steps=max_steps,
                )
                if scenario_has_hazard_opportunity(
                    network=network,
                    scenario=candidate,
                    profile_for_sampling=profile_for_sampling,
                    sampling_options=sampling_options,
                ):
                    scenarios.append(candidate)
                    accepted += 1
            if accepted < scenarios_per_ri:
                if not bool(sampling_options.get("allow_random_fallback", True)):
                    raise RuntimeError(
                        f"{ri_key} found {accepted}/{scenarios_per_ri} hazard-opportunity "
                        f"scenarios after {attempts} attempts."
                    )
                print(
                    f"WARNING: {ri_key} found {accepted}/{scenarios_per_ri} "
                    "hazard-opportunity scenarios; filling the remainder randomly."
                )
                while accepted < scenarios_per_ri:
                    sample = rng.sample(pool, num_deliveries + 1)
                    scenarios.append(
                        Scenario(
                            scenario_id=f"{scenario_prefix}_{ri_key}_{accepted:05d}",
                            ri_key=ri_key,
                            start=sample[0],
                            deliveries=tuple(sample[1:]),
                            max_steps=max_steps,
                        )
                    )
                    accepted += 1
            continue
        for idx in range(scenarios_per_ri):
            sample = rng.sample(pool, num_deliveries + 1)
            scenario_id = f"{scenario_prefix}_{ri_key}_{idx:05d}"
            scenarios.append(
                Scenario(
                    scenario_id=scenario_id,
                    ri_key=ri_key,
                    start=sample[0],
                    deliveries=tuple(sample[1:]),
                    max_steps=max_steps,
                )
            )
    return scenarios


def scenario_has_hazard_opportunity(
    network: ActivatedNetwork,
    scenario: Scenario,
    profile_for_sampling: str,
    sampling_options: dict[str, Any],
) -> bool:
    base_policy = str(sampling_options.get("opportunity_base_policy", "block_blind"))
    if base_policy == "passable_time":
        base = run_greedy_time_passable(network, scenario, profile_for_sampling)
    else:
        base = run_greedy_time_block_blind(network, scenario, profile_for_sampling)
    aware = run_greedy_hazard_aware(network, scenario, profile_for_sampling)
    if not base.success or not aware.success:
        return False
    risk_delta = base.common_risk_exposure - aware.common_risk_exposure
    time_delta = aware.travel_time_min - base.travel_time_min
    if base.common_risk_exposure < float(sampling_options.get("min_base_common_risk", 0.0)):
        return False
    if risk_delta < float(sampling_options.get("min_common_risk_delta", 5.0)):
        return False
    min_time_penalty = sampling_options.get("min_time_penalty")
    if min_time_penalty is not None and time_delta < float(min_time_penalty):
        return False
    max_time_penalty = sampling_options.get("max_time_penalty")
    if max_time_penalty is not None and time_delta > float(max_time_penalty):
        return False
    if bool(sampling_options.get("require_order_difference", False)):
        return base.visit_order != aware.visit_order
    return True


def optimal_order(network: ActivatedNetwork, start: str, deliveries: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
    deliveries = tuple(sorted(deliveries))

    @lru_cache(maxsize=None)
    def cost_to_go(current: str, remaining: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
        if not remaining:
            return 0.0, ()
        best_cost = float("inf")
        best_order: tuple[str, ...] = ()
        for target in remaining:
            path = network.path(current, target, weight="objective", passable=True)
            if path is None:
                continue
            rest = tuple(x for x in remaining if x != target)
            future_cost, future_order = cost_to_go(target, rest)
            total = path.objective + future_cost
            if total < best_cost:
                best_cost = total
                best_order = (target,) + future_order
        return best_cost, best_order

    return cost_to_go(start, deliveries)


def exact_action_cost(
    network: ActivatedNetwork,
    current: str,
    target: str,
    remaining: tuple[str, ...],
) -> float:
    path = network.path(current, target, weight="objective", passable=True)
    if path is None:
        return float("inf")
    rest = tuple(x for x in remaining if x != target)
    future_cost, _ = optimal_order(network, target, rest)
    return path.objective + future_cost


def feature_vector(
    bundle: GraphBundle,
    network: ActivatedNetwork,
    current: str,
    target: str,
    remaining: tuple[str, ...],
    num_deliveries: int,
) -> list[float]:
    cx, cy = bundle.xy_norm(current)
    tx, ty = bundle.xy_norm(target)
    dx = tx - cx
    dy = ty - cy
    euclidean = math.sqrt(dx * dx + dy * dy)
    rain_idx = max(0, int(network.rain.key.replace("RI", "")) - 1)
    rain_norm = rain_idx / 4.0

    path = network.path(current, target, weight="objective", passable=True)
    if path is None:
        path = PathMetrics((), 1e6, 1e6, 1e6, 1e3, 1e3, 1e3, 1e3, 10_000)

    hops = max(path.hops, 1)
    others = tuple(node for node in remaining if node != target)
    future_objectives: list[float] = []
    future_times: list[float] = []
    future_risks: list[float] = []
    for other in others:
        p = network.path(target, other, weight="objective", passable=True)
        if p is None:
            continue
        future_objectives.append(p.objective)
        future_times.append(p.travel_time_min)
        future_risks.append(p.risk_exposure / max(p.hops, 1))

    future_greedy = greedy_completion_metrics(network, target, others)
    candidate_greedy_completion = path.objective + future_greedy.objective

    current_objectives: list[tuple[float, str]] = []
    current_times: list[tuple[float, str]] = []
    for candidate in remaining:
        p_obj = network.path(current, candidate, weight="objective", passable=True)
        p_time = network.path(current, candidate, weight="travel_time", passable=True)
        if p_obj is not None:
            current_objectives.append((p_obj.objective, candidate))
        if p_time is not None:
            current_times.append((p_time.travel_time_min, candidate))

    obj_rank = rank_fraction(current_objectives, target)
    time_rank = rank_fraction(current_times, target)

    return [
        cx,
        cy,
        tx,
        ty,
        dx,
        dy,
        euclidean,
        len(remaining) / max(num_deliveries, 1),
        rain_norm,
        network.profile.eta_time,
        network.profile.hazard_lambda / 10.0,
        network.profile.w_flood,
        network.profile.w_landslide,
        math.log1p(path.objective) / 5.0,
        math.log1p(path.travel_time_min) / 5.0,
        math.log1p(path.distance_m / 1000.0),
        path.flood_exposure / hops,
        path.landslide_exposure / hops,
        path.risk_exposure / hops,
        path.hops / bundle.max_hops_scale,
        math.log1p(min(future_objectives) if future_objectives else 0.0) / 5.0,
        math.log1p(statistics.fmean(future_objectives) if future_objectives else 0.0) / 5.0,
        math.log1p(min(future_times) if future_times else 0.0) / 5.0,
        math.log1p(statistics.fmean(future_times) if future_times else 0.0) / 5.0,
        statistics.fmean(future_risks) if future_risks else 0.0,
        math.log1p(future_greedy.objective) / 5.0,
        math.log1p(future_greedy.travel_time_min) / 5.0,
        future_greedy.risk_exposure / max(future_greedy.hops, 1),
        future_greedy.hops / bundle.max_hops_scale,
        math.log1p(candidate_greedy_completion) / 5.0,
        obj_rank,
        time_rank,
    ]


def greedy_completion_metrics(
    network: ActivatedNetwork,
    current: str,
    remaining: tuple[str, ...],
) -> PathMetrics:
    if not remaining:
        return PathMetrics.empty(current)
    cursor = current
    todo = list(remaining)
    total = PathMetrics.empty(current)
    while todo:
        best: tuple[float, str, PathMetrics] | None = None
        for target in todo:
            path = network.path(cursor, target, weight="objective", passable=True)
            if path is None:
                continue
            if best is None or path.objective < best[0]:
                best = (path.objective, target, path)
        if best is None:
            return PathMetrics((), 1e6, 1e6, 1e6, 1e3, 1e3, 1e3, 1e3, 10_000)
        _, target, path = best
        total = add_metrics(total, path)
        cursor = target
        todo.remove(target)
    return total


def rank_fraction(costs: list[tuple[float, str]], target: str) -> float:
    if not costs:
        return 1.0
    ranked = sorted(costs, key=lambda item: item[0])
    if len(ranked) == 1:
        return 0.0
    for idx, (_, node) in enumerate(ranked):
        if node == target:
            return idx / (len(ranked) - 1)
    return 1.0


def collect_offline_transitions(
    bundle: GraphBundle,
    network: ActivatedNetwork,
    scenario: Scenario,
    num_deliveries: int,
) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    immediate_costs: list[float] = []
    dp_labels: list[float] = []
    done_flags: list[float] = []
    next_counts: list[int] = []
    next_rows: list[list[list[float]]] = []
    max_next_actions = max(0, num_deliveries - 1)
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def visit_state(current: str, remaining: tuple[str, ...]) -> None:
        remaining = tuple(sorted(remaining))
        key = (current, remaining)
        if key in seen or not remaining:
            return
        seen.add(key)
        for target in remaining:
            path = network.path(current, target, weight="objective", passable=True)
            if path is None:
                continue
            rest = tuple(x for x in remaining if x != target)
            label = exact_action_cost(network, current, target, remaining)
            if not math.isfinite(label):
                continue
            rows.append(feature_vector(bundle, network, current, target, remaining, num_deliveries))
            immediate_costs.append(path.objective)
            dp_labels.append(label)
            if not rest:
                done_flags.append(1.0)
                next_counts.append(0)
                next_rows.append([])
            else:
                candidate_rows: list[list[float]] = []
                for next_target in rest:
                    next_label = exact_action_cost(network, target, next_target, rest)
                    if not math.isfinite(next_label):
                        continue
                    candidate_rows.append(
                        feature_vector(bundle, network, target, next_target, rest, num_deliveries)
                    )
                if not candidate_rows:
                    # Should be rare if the current action is finite, but keep the transition valid.
                    done_flags.append(1.0)
                    next_counts.append(0)
                    next_rows.append([])
                else:
                    done_flags.append(0.0)
                    next_counts.append(len(candidate_rows))
                    next_rows.append(candidate_rows)
        for target in remaining:
            rest = tuple(x for x in remaining if x != target)
            visit_state(target, rest)

    visit_state(scenario.start, scenario.deliveries)

    curr_x = np.asarray(rows, dtype=np.float32)
    immediate_cost = np.asarray(immediate_costs, dtype=np.float32)
    dp_label = np.asarray(dp_labels, dtype=np.float32)
    done = np.asarray(done_flags, dtype=np.float32)
    next_count = np.asarray(next_counts, dtype=np.int64)
    next_x = np.zeros((len(rows), max_next_actions, len(FEATURE_NAMES)), dtype=np.float32)
    next_mask = np.zeros((len(rows), max_next_actions), dtype=np.float32)
    for idx, candidates in enumerate(next_rows):
        for j, feat in enumerate(candidates[:max_next_actions]):
            next_x[idx, j, :] = np.asarray(feat, dtype=np.float32)
            next_mask[idx, j] = 1.0
    return {
        "curr_x": curr_x,
        "immediate_cost": immediate_cost,
        "dp_label": dp_label,
        "done": done,
        "next_count": next_count,
        "next_x": next_x,
        "next_mask": next_mask,
    }


def offline_ddqn_finetune(
    model: CostToGoNet,
    checkpoint_meta: dict[str, Any],
    transitions: dict[str, np.ndarray],
    profile_key: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ddqn_cfg = dict(cfg.get("offline_ddqn", {}))
    if not bool(ddqn_cfg.get("enabled", False)):
        return {"enabled": False, "profile": profile_key, "rows": int(transitions["curr_x"].shape[0])}

    curr_x = transitions["curr_x"]
    immediate_cost = transitions["immediate_cost"]
    dp_label = transitions["dp_label"]
    done = transitions["done"]
    next_x = transitions["next_x"]
    next_mask = transitions["next_mask"]
    n_rows = int(curr_x.shape[0])
    if n_rows == 0:
        return {"enabled": True, "profile": profile_key, "rows": 0, "status": "no_rows"}

    label_mean = float(checkpoint_meta["label_mean"])
    label_std = float(checkpoint_meta["label_std"] if checkpoint_meta["label_std"] > 1e-9 else 1.0)
    use_anchor = bool(ddqn_cfg.get("use_dp_anchor_loss", True))
    anchor_alpha = float(ddqn_cfg.get("dp_anchor_alpha", 0.25))
    gamma = float(ddqn_cfg.get("gamma", 1.0))
    batch_size = int(ddqn_cfg.get("batch_size", 256))
    epochs = int(ddqn_cfg.get("epochs", 3))
    target_update_every = int(ddqn_cfg.get("target_update_every", 200))
    lr = float(ddqn_cfg.get("learning_rate", 0.0002))
    seed_offset = int(ddqn_cfg.get("seed_offset", 777))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(ddqn_cfg.get("weight_decay", 1e-4)))
    target_model = CostToGoNet(len(FEATURE_NAMES), list(checkpoint_meta["hidden_sizes"]))
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    model.train()

    mse = nn.MSELoss()
    indices = np.arange(n_rows)
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + seed_offset)
    step_count = 0
    epoch_stats: list[dict[str, float]] = []

    def to_raw(pred_norm: torch.Tensor) -> torch.Tensor:
        return pred_norm * label_std + label_mean

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        td_losses: list[float] = []
        anchor_losses: list[float] = []
        total_losses: list[float] = []
        for start in range(0, n_rows, batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = torch.from_numpy(curr_x[batch_idx])
            curr_pred_norm = model(xb)

            y_raw_list: list[float] = []
            for local_i, row_idx in enumerate(batch_idx):
                if done[row_idx] >= 0.5:
                    y_raw_list.append(float(immediate_cost[row_idx]))
                    continue
                valid_count = int(next_mask[row_idx].sum())
                next_feats = torch.from_numpy(next_x[row_idx, :valid_count, :])
                with torch.no_grad():
                    online_next_raw = to_raw(model(next_feats))
                    chosen_j = int(torch.argmin(online_next_raw).item())
                    target_next_raw = to_raw(target_model(next_feats))[chosen_j]
                    y_raw = float(immediate_cost[row_idx]) + gamma * float(target_next_raw.item())
                    y_raw_list.append(y_raw)
            y_raw = torch.tensor(y_raw_list, dtype=torch.float32)
            y_norm = (y_raw - label_mean) / label_std
            td_loss = mse(curr_pred_norm, y_norm)

            if use_anchor:
                dp_norm = torch.from_numpy((dp_label[batch_idx] - label_mean) / label_std)
                anchor_loss = mse(curr_pred_norm, dp_norm)
                loss = td_loss + anchor_alpha * anchor_loss
                anchor_losses.append(float(anchor_loss.item()))
            else:
                anchor_loss = None
                loss = td_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if ddqn_cfg.get("gradient_clip_norm") is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(ddqn_cfg["gradient_clip_norm"]))
            optimizer.step()
            step_count += 1
            if step_count % max(target_update_every, 1) == 0:
                target_model.load_state_dict(model.state_dict())

            td_losses.append(float(td_loss.item()))
            total_losses.append(float(loss.item()))

        target_model.load_state_dict(model.state_dict())
        epoch_stats.append({
            "epoch": epoch,
            "td_loss": statistics.fmean(td_losses) if td_losses else 0.0,
            "anchor_loss": statistics.fmean(anchor_losses) if anchor_losses else 0.0,
            "total_loss": statistics.fmean(total_losses) if total_losses else 0.0,
        })
        print(
            f"  {profile_key}: offline DDQN epoch {epoch:03d} td_mse={epoch_stats[-1]['td_loss']:.5f} "
            f"anchor_mse={epoch_stats[-1]['anchor_loss']:.5f} total={epoch_stats[-1]['total_loss']:.5f}"
        )

    checkpoint_meta["offline_ddqn"] = {
        "enabled": True,
        "use_dp_anchor_loss": use_anchor,
        "dp_anchor_alpha": anchor_alpha,
        "gamma": gamma,
        "epochs": epochs,
        "batch_size": batch_size,
        "target_update_every": target_update_every,
        "learning_rate": lr,
        "rows": n_rows,
        "seconds": time.perf_counter() - t0,
        "epoch_stats": epoch_stats,
    }
    return checkpoint_meta["offline_ddqn"]


def collect_macro_candidates(
    bundle: GraphBundle,
    network: ActivatedNetwork,
    current: str,
    remaining: tuple[str, ...],
    num_deliveries: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    remaining = tuple(sorted(remaining))
    for target in remaining:
        path = network.path(current, target, weight="objective", passable=True)
        if path is None:
            continue
        feat = feature_vector(bundle, network, current, target, remaining, num_deliveries)
        rest = tuple(x for x in remaining if x != target)
        next_features: list[list[float]] = []
        if rest:
            for next_target in rest:
                next_path = network.path(target, next_target, weight="objective", passable=True)
                if next_path is None:
                    continue
                next_features.append(feature_vector(bundle, network, target, next_target, rest, num_deliveries))
        candidates.append({
            "target": target,
            "feature": feat,
            "immediate_cost": float(path.objective),
            "rest": rest,
            "done": not rest or not next_features,
            "next_features": next_features,
        })
    return candidates


def online_ddqn_train(
    model: CostToGoNet,
    checkpoint_meta: dict[str, Any],
    bundle: GraphBundle,
    networks: dict[tuple[str, str], ActivatedNetwork],
    scenarios: list[Scenario],
    profile_key: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    online_cfg = dict(cfg.get("online_ddqn", {}))
    if not bool(online_cfg.get("enabled", False)):
        return {"enabled": False, "profile": profile_key, "episodes": 0}

    label_mean = float(checkpoint_meta.get("label_mean", 0.0))
    label_std = float(checkpoint_meta.get("label_std", 1.0) if checkpoint_meta.get("label_std", 1.0) > 1e-9 else 1.0)
    gamma = float(online_cfg.get("gamma", 1.0))
    batch_size = int(online_cfg.get("batch_size", 128))
    epochs = int(online_cfg.get("epochs", 8))
    lr = float(online_cfg.get("learning_rate", 0.0002))
    target_update_every = int(online_cfg.get("target_update_every", 200))
    replay_capacity = int(online_cfg.get("replay_capacity", 50000))
    min_replay_to_train = int(online_cfg.get("min_replay_to_train", 512))
    updates_per_step = int(online_cfg.get("updates_per_step", 1))
    epsilon_start = float(online_cfg.get("epsilon_start", 0.20))
    epsilon_end = float(online_cfg.get("epsilon_end", 0.02))
    epsilon_decay_epochs = max(1, int(online_cfg.get("epsilon_decay_epochs", max(1, epochs // 2))))
    seed_offset = int(online_cfg.get("seed_offset", 1337))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(online_cfg.get("weight_decay", 1e-4)))
    target_model = CostToGoNet(len(FEATURE_NAMES), list(checkpoint_meta["hidden_sizes"]))
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    model.train()
    mse = nn.MSELoss()
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + seed_offset)
    replay: list[dict[str, Any]] = []
    grad_step = 0
    epoch_stats: list[dict[str, float]] = []
    num_deliveries = int(cfg["scenario"]["num_deliveries"])

    def to_raw(pred_norm: torch.Tensor) -> torch.Tensor:
        return pred_norm * label_std + label_mean

    def sample_batch(batch_n: int) -> list[dict[str, Any]]:
        idx = rng.choice(len(replay), size=batch_n, replace=False)
        return [replay[int(i)] for i in idx]

    def ddqn_update(batch: list[dict[str, Any]]) -> float:
        nonlocal grad_step
        xb = torch.tensor(np.asarray([item["curr_x"] for item in batch], dtype=np.float32))
        curr_pred_norm = model(xb)
        y_raw_list: list[float] = []
        for item in batch:
            if item["done"] or not item["next_x"]:
                y_raw_list.append(float(item["immediate_cost"]))
                continue
            next_feats = torch.tensor(np.asarray(item["next_x"], dtype=np.float32))
            with torch.no_grad():
                online_next_raw = to_raw(model(next_feats))
                chosen_j = int(torch.argmin(online_next_raw).item())
                target_next_raw = to_raw(target_model(next_feats))[chosen_j]
                y_raw = float(item["immediate_cost"]) + gamma * float(target_next_raw.item())
                y_raw_list.append(y_raw)
        y_raw = torch.tensor(y_raw_list, dtype=torch.float32)
        y_norm = (y_raw - label_mean) / label_std
        loss = mse(curr_pred_norm, y_norm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if online_cfg.get("gradient_clip_norm") is not None:
            nn.utils.clip_grad_norm_(model.parameters(), float(online_cfg["gradient_clip_norm"]))
        optimizer.step()
        grad_step += 1
        if grad_step % max(target_update_every, 1) == 0:
            target_model.load_state_dict(model.state_dict())
        return float(loss.item())

    scenario_order = list(range(len(scenarios)))
    t0 = time.perf_counter()
    episodes = 0
    for epoch in range(1, epochs + 1):
        rng.shuffle(scenario_order)
        td_losses: list[float] = []
        epsilon_progress = min(1.0, (epoch - 1) / float(epsilon_decay_epochs))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * epsilon_progress
        for sidx in scenario_order:
            scenario = scenarios[sidx]
            network = networks[(scenario.ri_key, profile_key)]
            current = scenario.start
            remaining = tuple(sorted(scenario.deliveries))
            while remaining:
                candidates = collect_macro_candidates(bundle, network, current, remaining, num_deliveries)
                if not candidates:
                    break
                if rng.random() < epsilon:
                    choice_idx = int(rng.integers(0, len(candidates)))
                else:
                    cand_x = torch.tensor(np.asarray([c["feature"] for c in candidates], dtype=np.float32))
                    with torch.no_grad():
                        pred_raw = to_raw(model(cand_x)).cpu().numpy()
                    choice_idx = int(np.argmin(pred_raw))
                chosen = candidates[choice_idx]
                replay.append({
                    "curr_x": np.asarray(chosen["feature"], dtype=np.float32),
                    "immediate_cost": float(chosen["immediate_cost"]),
                    "done": bool(chosen["done"]),
                    "next_x": [np.asarray(f, dtype=np.float32) for f in chosen["next_features"]],
                })
                if len(replay) > replay_capacity:
                    replay.pop(0)
                if len(replay) >= min_replay_to_train:
                    for _ in range(max(updates_per_step, 1)):
                        batch = sample_batch(min(batch_size, len(replay)))
                        td_losses.append(ddqn_update(batch))
                current = str(chosen["target"])
                remaining = tuple(sorted(chosen["rest"]))
            episodes += 1
        target_model.load_state_dict(model.state_dict())
        stat = {
            "epoch": epoch,
            "epsilon": float(epsilon),
            "td_loss": statistics.fmean(td_losses) if td_losses else 0.0,
            "replay_size": float(len(replay)),
            "episodes": float(episodes),
        }
        epoch_stats.append(stat)
        print(
            f"  {profile_key}: online DDQN epoch {epoch:03d} epsilon={stat['epsilon']:.3f} "
            f"td_mse={stat['td_loss']:.5f} replay={int(stat['replay_size'])}"
        )

    checkpoint_meta["online_ddqn"] = {
        "enabled": True,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "gamma": gamma,
        "target_update_every": target_update_every,
        "replay_capacity": replay_capacity,
        "min_replay_to_train": min_replay_to_train,
        "updates_per_step": updates_per_step,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_epochs": epsilon_decay_epochs,
        "episodes": episodes,
        "seconds": time.perf_counter() - t0,
        "epoch_stats": epoch_stats,
    }
    return checkpoint_meta["online_ddqn"]


def collect_training_rows(
    bundle: GraphBundle,
    network: ActivatedNetwork,
    scenario: Scenario,
    num_deliveries: int,
) -> tuple[list[list[float]], list[float]]:
    rows: list[list[float]] = []
    labels: list[float] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def visit_state(current: str, remaining: tuple[str, ...]) -> None:
        remaining = tuple(sorted(remaining))
        key = (current, remaining)
        if key in seen or not remaining:
            return
        seen.add(key)
        for target in remaining:
            label = exact_action_cost(network, current, target, remaining)
            if not math.isfinite(label):
                continue
            rows.append(feature_vector(bundle, network, current, target, remaining, num_deliveries))
            labels.append(label)
        for target in remaining:
            rest = tuple(x for x in remaining if x != target)
            visit_state(target, rest)

    visit_state(scenario.start, scenario.deliveries)
    return rows, labels


def train_profile_model(
    bundle: GraphBundle,
    networks: dict[tuple[str, str], ActivatedNetwork],
    scenarios: list[Scenario],
    profile_key: str,
    cfg: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    training_cfg = cfg["training"]
    num_deliveries = int(cfg["scenario"]["num_deliveries"])
    online_cfg = dict(cfg.get("online_ddqn", {}))
    online_enabled = bool(online_cfg.get("enabled", False))
    init_mode = str(online_cfg.get("init_mode", "full_pretrain")) if online_enabled else "full_pretrain"

    rows: list[list[float]] = []
    labels: list[float] = []
    transition_parts: list[dict[str, np.ndarray]] = []
    t0 = time.perf_counter()
    need_supervised_rows = init_mode != "scratch"
    for idx, scenario in enumerate(scenarios, start=1):
        network = networks[(scenario.ri_key, profile_key)]
        if need_supervised_rows:
            x, y = collect_training_rows(bundle, network, scenario, num_deliveries)
            rows.extend(x)
            labels.extend(y)
            if idx % 200 == 0:
                print(f"  {profile_key}: collected {idx}/{len(scenarios)} scenarios -> {len(rows)} action rows")
        if bool(cfg.get("offline_ddqn", {}).get("enabled", False)):
            transition_parts.append(collect_offline_transitions(bundle, network, scenario, num_deliveries))

    x_arr = None
    y_arr = None
    y_norm = None
    val_idx = np.asarray([], dtype=np.int64)
    train_idx = np.asarray([], dtype=np.int64)
    val_fraction = float(training_cfg.get("validation_fraction", 0.15))
    if need_supervised_rows:
        if not rows:
            raise RuntimeError(f"No training rows generated for profile {profile_key}.")
        x_arr = np.asarray(rows, dtype=np.float32)
        y_arr = np.asarray(labels, dtype=np.float32)
        label_mean = float(y_arr.mean())
        label_std = float(y_arr.std() if y_arr.std() > 1e-6 else 1.0)
        y_norm = (y_arr - label_mean) / label_std

        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        indices = rng.permutation(len(x_arr))
        val_count = max(1, int(len(indices) * val_fraction))
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
    else:
        label_mean = 0.0
        label_std = 1.0

    model = CostToGoNet(len(FEATURE_NAMES), list(training_cfg.get("hidden_sizes", [192, 96, 48])))
    best_val = float("inf")
    best_epoch = 0
    pretrain_epochs_run = 0
    pretrain_max_epochs = 0

    pretrain_model_path = None
    if init_mode != "scratch":
        train_ds = TensorDataset(
            torch.from_numpy(x_arr[train_idx]),
            torch.from_numpy(y_norm[train_idx]),
        )
        val_x = torch.from_numpy(x_arr[val_idx])
        val_y = torch.from_numpy(y_norm[val_idx])

        batch_size = int(training_cfg.get("batch_size", 256))
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg.get("learning_rate", 0.001)),
            weight_decay=1e-4,
        )
        loss_fn = nn.MSELoss()

        best_state = None
        no_improve = 0
        max_epochs = int(training_cfg.get("epochs", 90))
        if online_enabled and init_mode == "light_pretrain":
            max_epochs = min(max_epochs, int(online_cfg.get("warm_start_epochs", 8)))
        pretrain_max_epochs = max_epochs
        patience = int(training_cfg.get("early_stop_patience", 18))
        if online_enabled and init_mode == "light_pretrain":
            patience = min(patience, max(2, int(online_cfg.get("warm_start_patience", 4))))

        for epoch in range(1, max_epochs + 1):
            pretrain_epochs_run = epoch
            model.train()
            train_losses: list[float] = []
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_loss = float(loss_fn(val_pred, val_y).item())
            train_loss = statistics.fmean(train_losses)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
                print(
                    f"  {profile_key}: epoch {epoch:03d} train_mse={train_loss:.5f} "
                    f"val_mse={val_loss:.5f} best={best_val:.5f}"
                )
            if no_improve >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        batch_size = int(training_cfg.get("batch_size", 256))

    checkpoint_meta = {
        "profile": profile_key,
        "feature_names": FEATURE_NAMES,
        "hidden_sizes": list(training_cfg.get("hidden_sizes", [192, 96, 48])),
        "label_mean": label_mean,
        "label_std": label_std,
        "num_rows": len(rows),
        "best_val_mse": best_val if math.isfinite(best_val) else None,
        "best_epoch": best_epoch,
        "config": cfg,
        "pretraining": {
            "mode": init_mode if online_enabled else "full_pretrain",
            "epochs": pretrain_epochs_run,
            "configured_epochs": 0 if init_mode == "scratch" else int(training_cfg.get("epochs", 90)),
            "max_epochs_after_mode_overrides": pretrain_max_epochs,
            "batch_size": batch_size,
            "learning_rate": float(training_cfg.get("learning_rate", 0.001)),
            "validation_fraction": val_fraction,
            "early_stop_patience": int(training_cfg.get("early_stop_patience", 18)),
        },
    }

    model_dir = artifact_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pretrain_model_path = model_dir / f"macro_dqn_{profile_key}_pretrain.pt"
    torch.save({**checkpoint_meta, "state_dict": model.state_dict()}, pretrain_model_path)

    ddqn_summary = {"enabled": False}
    if bool(cfg.get("offline_ddqn", {}).get("enabled", False)) and transition_parts:
        transitions = {
            key: np.concatenate([part[key] for part in transition_parts], axis=0)
            for key in transition_parts[0].keys()
        }
        ddqn_summary = offline_ddqn_finetune(model, checkpoint_meta, transitions, profile_key, cfg)

    online_ddqn_summary = {"enabled": False}
    if online_enabled:
        online_ddqn_summary = online_ddqn_train(model, checkpoint_meta, bundle, networks, scenarios, profile_key, cfg)

    model_path = model_dir / f"macro_dqn_{profile_key}.pt"
    torch.save({**checkpoint_meta, "state_dict": model.state_dict()}, model_path)

    return {
        "profile": profile_key,
        "rows": len(rows),
        "label_mean": label_mean,
        "label_std": label_std,
        "best_val_mse": best_val if math.isfinite(best_val) else None,
        "best_epoch": best_epoch,
        "model_path": str(model_path),
        "pretrain_model_path": str(pretrain_model_path),
        "offline_ddqn_enabled": int(bool(ddqn_summary.get("enabled", False))),
        "offline_ddqn_use_dp_anchor": int(bool(ddqn_summary.get("use_dp_anchor_loss", False))),
        "offline_ddqn_td_epochs": int(ddqn_summary.get("epochs", 0) or 0),
        "offline_ddqn_seconds": float(ddqn_summary.get("seconds", 0.0) or 0.0),
        "online_ddqn_enabled": int(bool(online_ddqn_summary.get("enabled", False))),
        "online_ddqn_init_mode": str(online_cfg.get("init_mode", "full_pretrain")) if online_enabled else "none",
        "online_ddqn_epochs": int(online_ddqn_summary.get("epochs", 0) or 0),
        "online_ddqn_seconds": float(online_ddqn_summary.get("seconds", 0.0) or 0.0),
        "seconds": time.perf_counter() - t0,
    }


def load_model(model_path: Path) -> tuple[CostToGoNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint["feature_names"] != FEATURE_NAMES:
        raise RuntimeError(f"Feature mismatch in {model_path}.")
    model = CostToGoNet(len(FEATURE_NAMES), list(checkpoint["hidden_sizes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_cost(
    model: CostToGoNet,
    checkpoint: dict[str, Any],
    features: list[list[float]],
) -> np.ndarray:
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        pred_norm = model(x).cpu().numpy()
    return pred_norm * float(checkpoint["label_std"]) + float(checkpoint["label_mean"])


def add_metrics(total: PathMetrics, part: PathMetrics) -> PathMetrics:
    if not total.path:
        full_path = part.path
    elif len(part.path) <= 1:
        full_path = total.path
    else:
        full_path = total.path + part.path[1:]
    return PathMetrics(
        path=full_path,
        objective=total.objective + part.objective,
        travel_time_min=total.travel_time_min + part.travel_time_min,
        distance_m=total.distance_m + part.distance_m,
        flood_exposure=total.flood_exposure + part.flood_exposure,
        landslide_exposure=total.landslide_exposure + part.landslide_exposure,
        blockage_exposure=total.blockage_exposure + part.blockage_exposure,
        risk_exposure=total.risk_exposure + part.risk_exposure,
        hops=total.hops + part.hops,
    )


def finish_route(
    scenario: Scenario,
    profile_key: str,
    algorithm: str,
    success: bool,
    reason: str,
    delivered: int,
    metrics: PathMetrics,
    order: list[str],
    replans: int,
    t0: float,
    common_hazard_weight: float = 0.5,
) -> RouteResult:
    return RouteResult(
        scenario_id=scenario.scenario_id,
        ri_key=scenario.ri_key,
        profile=profile_key,
        algorithm=algorithm,
        success=success,
        failure_reason=reason,
        delivered=delivered,
        objective=metrics.objective,
        travel_time_min=metrics.travel_time_min,
        distance_m=metrics.distance_m,
        flood_exposure=metrics.flood_exposure,
        landslide_exposure=metrics.landslide_exposure,
        blockage_exposure=metrics.blockage_exposure,
        risk_exposure=metrics.risk_exposure,
        common_risk_exposure=(
            common_hazard_weight * (metrics.flood_exposure + metrics.landslide_exposure)
            + metrics.blockage_exposure
        ),
        hops=metrics.hops,
        replan_count=replans,
        visit_order="|".join(order),
        wall_time_ms=(time.perf_counter() - t0) * 1000.0,
    )


def run_order_policy(
    network: ActivatedNetwork,
    scenario: Scenario,
    profile_key: str,
    algorithm: str,
    order: tuple[str, ...],
) -> RouteResult:
    t0 = time.perf_counter()
    current = scenario.start
    metrics = PathMetrics.empty(current)
    visit_order: list[str] = []
    for target in order:
        path = network.path(current, target, weight="objective", passable=True)
        if path is None:
            return finish_route(
                scenario, profile_key, algorithm, False, "no_route",
                len(visit_order), metrics, visit_order, 0, t0, network.common_hazard_weight
            )
        metrics = add_metrics(metrics, path)
        visit_order.append(target)
        current = target
        if metrics.hops > scenario.max_steps:
            return finish_route(
                scenario, profile_key, algorithm, False, "timeout",
                len(visit_order), metrics, visit_order, 0, t0, network.common_hazard_weight
            )
    return finish_route(
        scenario, profile_key, algorithm, True, "", len(visit_order), metrics, visit_order, 0, t0,
        network.common_hazard_weight
    )


def run_greedy_hazard_aware(
    network: ActivatedNetwork,
    scenario: Scenario,
    profile_key: str,
) -> RouteResult:
    t0 = time.perf_counter()
    current = scenario.start
    remaining = list(scenario.deliveries)
    metrics = PathMetrics.empty(current)
    order: list[str] = []
    while remaining:
        best: tuple[float, str, PathMetrics] | None = None
        for target in remaining:
            path = network.path(current, target, weight="objective", passable=True)
            if path is None:
                continue
            if best is None or path.objective < best[0]:
                best = (path.objective, target, path)
        if best is None:
            return finish_route(
                scenario, profile_key, "Greedy-HazardAware", False, "no_route",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )
        _, target, path = best
        metrics = add_metrics(metrics, path)
        order.append(target)
        remaining.remove(target)
        current = target
        if metrics.hops > scenario.max_steps:
            return finish_route(
                scenario, profile_key, "Greedy-HazardAware", False, "timeout",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )
    return finish_route(
        scenario, profile_key, "Greedy-HazardAware", True, "", len(order), metrics, order, 0, t0,
        network.common_hazard_weight
    )


def run_greedy_time_block_blind(
    network: ActivatedNetwork,
    scenario: Scenario,
    profile_key: str,
) -> RouteResult:
    t0 = time.perf_counter()
    current = scenario.start
    remaining = list(scenario.deliveries)
    metrics = PathMetrics.empty(current)
    order: list[str] = []
    while remaining:
        best: tuple[float, str, PathMetrics] | None = None
        for target in remaining:
            path = network.path(current, target, weight="base_time", passable=False)
            if path is None:
                continue
            plan_cost = sum(
                float(network.full_graph[u][v]["base_time"])
                for u, v in zip(path.path, path.path[1:])
            )
            if best is None or plan_cost < best[0]:
                best = (plan_cost, target, path)
        if best is None:
            return finish_route(
                scenario, profile_key, "Greedy-Time-Base", False, "no_route",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )
        _, target, plan = best
        executed = network.execute_planned_path(plan.path, passable=False, weight="base_time")
        if executed is None:
            return finish_route(
                scenario, profile_key, "Greedy-Time-Base", False, "blocked_edge",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )
        metrics = add_metrics(metrics, executed)
        order.append(target)
        remaining.remove(target)
        current = target
        if metrics.hops > scenario.max_steps:
            return finish_route(
                scenario, profile_key, "Greedy-Time-Base", False, "timeout",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )
    return finish_route(
        scenario, profile_key, "Greedy-Time-Base", True, "", len(order), metrics, order, 0, t0,
        network.common_hazard_weight
    )


def run_greedy_time_replan(
    network: ActivatedNetwork,
    scenario: Scenario,
    profile_key: str,
) -> RouteResult:
    t0 = time.perf_counter()
    current = scenario.start
    remaining = list(scenario.deliveries)
    metrics = PathMetrics.empty(current)
    order: list[str] = []
    while remaining:
        best: tuple[float, str, PathMetrics] | None = None
        for target in remaining:
            path = network.path(current, target, weight="travel_time", passable=False)
            if path is None:
                continue
            plan_cost = path.travel_time_min
            if best is None or plan_cost < best[0]:
                best = (plan_cost, target, path)
        if best is None:
            return finish_route(
                scenario, profile_key, "Greedy-Time-SoftPenalty", False, "no_route",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )

        _, target, plan = best
        executed = network.execute_planned_path(plan.path, passable=False, weight="travel_time")
        if executed is None:
            return finish_route(
                scenario, profile_key, "Greedy-Time-SoftPenalty", False, "blocked_edge",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )
        metrics = add_metrics(metrics, executed)
        order.append(target)
        remaining.remove(target)
        current = target
        if metrics.hops > scenario.max_steps:
            return finish_route(
                scenario, profile_key, "Greedy-Time-SoftPenalty", False, "timeout",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )

    return finish_route(
        scenario, profile_key, "Greedy-Time-SoftPenalty", True, "", len(order), metrics, order, 0, t0,
        network.common_hazard_weight
    )


def run_greedy_time_passable(
    network: ActivatedNetwork,
    scenario: Scenario,
    profile_key: str,
) -> RouteResult:
    t0 = time.perf_counter()
    current = scenario.start
    remaining = list(scenario.deliveries)
    metrics = PathMetrics.empty(current)
    order: list[str] = []
    while remaining:
        best: tuple[float, str, PathMetrics] | None = None
        for target in remaining:
            path = network.path(current, target, weight="travel_time", passable=True)
            if path is None:
                continue
            if best is None or path.travel_time_min < best[0]:
                best = (path.travel_time_min, target, path)
        if best is None:
            return finish_route(
                scenario, profile_key, "Greedy-Time-Passable", False, "no_route",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )

        _, target, path = best
        metrics = add_metrics(metrics, path)
        order.append(target)
        remaining.remove(target)
        current = target
        if metrics.hops > scenario.max_steps:
            return finish_route(
                scenario, profile_key, "Greedy-Time-Passable", False, "timeout",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )

    return finish_route(
        scenario, profile_key, "Greedy-Time-Passable", True, "", len(order), metrics, order, 0, t0,
        network.common_hazard_weight
    )


def run_macro_dqn(
    bundle: GraphBundle,
    network: ActivatedNetwork,
    scenario: Scenario,
    profile_key: str,
    model: CostToGoNet,
    checkpoint: dict[str, Any],
    num_deliveries: int,
) -> RouteResult:
    t0 = time.perf_counter()
    current = scenario.start
    remaining = list(scenario.deliveries)
    metrics = PathMetrics.empty(current)
    order: list[str] = []
    while remaining:
        features = [
            feature_vector(bundle, network, current, target, tuple(remaining), num_deliveries)
            for target in remaining
        ]
        predicted_costs = predict_cost(model, checkpoint, features)
        ranked = sorted(zip(predicted_costs, remaining), key=lambda item: item[0])
        chosen = None
        chosen_path = None
        for _, target in ranked:
            path = network.path(current, target, weight="objective", passable=True)
            if path is not None:
                chosen = target
                chosen_path = path
                break
        if chosen is None or chosen_path is None:
            return finish_route(
                scenario, profile_key, "Macro-DQN", False, "no_route",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )
        metrics = add_metrics(metrics, chosen_path)
        order.append(chosen)
        remaining.remove(chosen)
        current = chosen
        if metrics.hops > scenario.max_steps:
            return finish_route(
                scenario, profile_key, "Macro-DQN", False, "timeout",
                len(order), metrics, order, 0, t0, network.common_hazard_weight
            )
    return finish_route(
        scenario, profile_key, "Macro-DQN", True, "", len(order), metrics, order, 0, t0,
        network.common_hazard_weight
    )


def evaluate(
    bundle: GraphBundle,
    networks: dict[tuple[str, str], ActivatedNetwork],
    scenarios: list[Scenario],
    profile_keys: list[str],
    cfg: dict[str, Any],
    artifact_dir: Path,
    eval_cfg: dict[str, Any] | None = None,
) -> list[RouteResult]:
    num_deliveries = int(cfg["scenario"]["num_deliveries"])
    model_source_dir = cfg.get("outputs", {}).get("model_source_dir")
    model_dir = resolve_path(model_source_dir) if model_source_dir else artifact_dir / "models"
    eval_cfg = eval_cfg or {}
    stochastic_per_scenario = bool(
        eval_cfg.get(
            "stochastic_network_per_scenario",
            cfg["hazard"].get("stochastic_evaluation", {}).get("per_scenario", False),
        )
    )
    rains = parse_rain_levels(cfg) if stochastic_per_scenario else {}
    profiles = parse_profiles(cfg) if stochastic_per_scenario else {}
    models: dict[str, tuple[CostToGoNet, dict[str, Any]]] = {}
    for profile_key in profile_keys:
        model_path = model_dir / f"macro_dqn_{profile_key}.pt"
        models[profile_key] = load_model(model_path)

    results: list[RouteResult] = []
    if stochastic_per_scenario:
        for idx, scenario in enumerate(scenarios, start=1):
            scenario_networks = make_networks(
                bundle=bundle,
                rains=rains,
                profiles=profiles,
                cfg=cfg,
                ri_keys=[scenario.ri_key],
                profile_keys=profile_keys,
                seed_offset=stable_seed_offset(
                    eval_cfg.get("seed", cfg.get("seed", 0)),
                    scenario.scenario_id,
                ),
            )
            for profile_key in profile_keys:
                model, checkpoint = models[profile_key]
                network = scenario_networks[(scenario.ri_key, profile_key)]
                results.append(run_greedy_time_block_blind(network, scenario, profile_key))
                results.append(run_greedy_time_replan(network, scenario, profile_key))
                results.append(run_greedy_hazard_aware(network, scenario, profile_key))
                _, order = optimal_order(network, scenario.start, scenario.deliveries)
                results.append(run_order_policy(network, scenario, profile_key, "Macro-DP-Oracle", order))
                results.append(
                    run_macro_dqn(bundle, network, scenario, profile_key, model, checkpoint, num_deliveries)
                )
            if idx % 200 == 0:
                print(f"  stochastic: evaluated {idx}/{len(scenarios)} scenarios")
        return results

    for profile_key in profile_keys:
        print(f"Evaluating profile: {profile_key}")
        model, checkpoint = models[profile_key]
        for idx, scenario in enumerate(scenarios, start=1):
            network = networks[(scenario.ri_key, profile_key)]
            results.append(run_greedy_time_block_blind(network, scenario, profile_key))
            results.append(run_greedy_time_replan(network, scenario, profile_key))
            results.append(run_greedy_hazard_aware(network, scenario, profile_key))
            _, order = optimal_order(network, scenario.start, scenario.deliveries)
            results.append(run_order_policy(network, scenario, profile_key, "Macro-DP-Oracle", order))
            results.append(
                run_macro_dqn(bundle, network, scenario, profile_key, model, checkpoint, num_deliveries)
            )
            if idx % 200 == 0:
                print(f"  {profile_key}: evaluated {idx}/{len(scenarios)} scenarios")
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results: list[RouteResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[RouteResult]] = defaultdict(list)
    for result in results:
        grouped[(result.profile, result.ri_key, result.algorithm)].append(result)

    rows: list[dict[str, Any]] = []
    for (profile, ri_key, algorithm), group in sorted(grouped.items()):
        success = [r for r in group if r.success]
        rows.append(
            {
                "profile": profile,
                "ri_key": ri_key,
                "algorithm": algorithm,
                "n": len(group),
                "success_rate": round(len(success) / max(len(group), 1), 6),
                "avg_delivered": round(statistics.fmean(r.delivered for r in group), 6),
                "avg_objective_success": mean_or_blank(r.objective for r in success),
                "avg_time_success": mean_or_blank(r.travel_time_min for r in success),
                "avg_distance_m_success": mean_or_blank(r.distance_m for r in success),
                "avg_risk_success": mean_or_blank(r.risk_exposure for r in success),
                "avg_common_risk_success": mean_or_blank(r.common_risk_exposure for r in success),
                "avg_flood_success": mean_or_blank(r.flood_exposure for r in success),
                "avg_landslide_success": mean_or_blank(r.landslide_exposure for r in success),
                "avg_blockage_success": mean_or_blank(r.blockage_exposure for r in success),
                "avg_hops_success": mean_or_blank(r.hops for r in success),
                "avg_replans": round(statistics.fmean(r.replan_count for r in group), 6),
                "avg_runtime_ms": round(statistics.fmean(r.wall_time_ms for r in group), 6),
            }
        )
    return rows


def write_deltas(summary_rows: list[dict[str, Any]], path: Path) -> None:
    index = {
        (row["profile"], row["ri_key"], row["algorithm"]): row
        for row in summary_rows
    }
    baselines = ["Greedy-Time-Base", "Greedy-Time-SoftPenalty", "Greedy-HazardAware"]
    rows: list[dict[str, Any]] = []
    for (profile, ri_key, algorithm), dqn_row in index.items():
        if algorithm != "Macro-DQN":
            continue
        for baseline in baselines:
            base = index.get((profile, ri_key, baseline))
            if not base:
                continue
            rows.append(
                {
                    "profile": profile,
                    "ri_key": ri_key,
                    "baseline": baseline,
                    "dqn_success_rate": dqn_row["success_rate"],
                    "baseline_success_rate": base["success_rate"],
                    "success_rate_delta": round(
                        float(dqn_row["success_rate"]) - float(base["success_rate"]), 6
                    ),
                    "dqn_objective": dqn_row["avg_objective_success"],
                    "baseline_objective": base["avg_objective_success"],
                    "objective_delta_success": numeric_delta(
                        dqn_row["avg_objective_success"], base["avg_objective_success"]
                    ),
                    "dqn_risk": dqn_row["avg_risk_success"],
                    "baseline_risk": base["avg_risk_success"],
                    "risk_delta_success": numeric_delta(
                        dqn_row["avg_risk_success"], base["avg_risk_success"]
                    ),
                    "dqn_common_risk": dqn_row["avg_common_risk_success"],
                    "baseline_common_risk": base["avg_common_risk_success"],
                    "common_risk_delta_success": numeric_delta(
                        dqn_row["avg_common_risk_success"], base["avg_common_risk_success"]
                    ),
                    "dqn_time": dqn_row["avg_time_success"],
                    "baseline_time": base["avg_time_success"],
                    "time_delta_success": numeric_delta(
                        dqn_row["avg_time_success"], base["avg_time_success"]
                    ),
                    "dqn_blockage": dqn_row["avg_blockage_success"],
                    "baseline_blockage": base["avg_blockage_success"],
                    "blockage_delta_success": numeric_delta(
                        dqn_row["avg_blockage_success"], base["avg_blockage_success"]
                    ),
                }
            )
    write_csv(path, rows)


def write_profile_specialization(summary_rows: list[dict[str, Any]], path: Path) -> None:
    dqn_rows = [
        row for row in summary_rows
        if row["algorithm"] == "Macro-DQN"
    ]
    by_ri: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in dqn_rows:
        by_ri[str(row["ri_key"])][str(row["profile"])] = row

    rows: list[dict[str, Any]] = []
    for ri_key, profiles in sorted(by_ri.items()):
        row: dict[str, Any] = {"ri_key": ri_key}
        for profile in sorted(profiles):
            source = profiles[profile]
            row[f"{profile}_objective"] = source["avg_objective_success"]
            row[f"{profile}_time"] = source["avg_time_success"]
            row[f"{profile}_common_risk"] = source["avg_common_risk_success"]
            row[f"{profile}_success_rate"] = source["success_rate"]
        if "safe" in profiles and "fast" in profiles:
            row["safe_minus_fast_common_risk"] = numeric_delta(
                profiles["safe"]["avg_common_risk_success"],
                profiles["fast"]["avg_common_risk_success"],
            )
            row["safe_minus_fast_time"] = numeric_delta(
                profiles["safe"]["avg_time_success"],
                profiles["fast"]["avg_time_success"],
            )
            row["safe_lower_common_risk_than_fast"] = int(
                float(profiles["safe"]["avg_common_risk_success"])
                < float(profiles["fast"]["avg_common_risk_success"])
            )
            row["fast_faster_than_safe"] = int(
                float(profiles["fast"]["avg_time_success"])
                < float(profiles["safe"]["avg_time_success"])
            )
        rows.append(row)
    write_csv(path, rows)


def numeric_delta(a: Any, b: Any) -> Any:
    if a == "" or b == "":
        return ""
    return round(float(a) - float(b), 6)


def mean_or_blank(values: Iterable[float]) -> Any:
    vals = list(values)
    if not vals:
        return ""
    return round(statistics.fmean(vals), 6)


def graph_report(
    bundle: GraphBundle,
    networks: dict[tuple[str, str], ActivatedNetwork],
    ri_keys: list[str],
    sampling_profile: str,
) -> dict[str, Any]:
    graph = bundle.graph
    undirected_edges = {canonical_edge(u, v) for u, v in graph.edges()}
    degree = dict(graph.to_undirected().degree())
    edge_values = {
        "flood_hazard": [],
        "landslide_hazard": [],
        "combined_hazard": [],
        "base_time": [],
        "length_m": [],
    }
    for u, v in undirected_edges:
        data = graph[u][v] if graph.has_edge(u, v) else graph[v][u]
        for key in edge_values:
            edge_values[key].append(float(data.get(key, 0.0)))
    report = {
        "nodes": graph.number_of_nodes(),
        "directed_edges": graph.number_of_edges(),
        "undirected_edges": len(undirected_edges),
        "degree_avg": statistics.fmean(degree.values()),
        "degree_max": max(degree.values()),
        "mean_edge_time_min": bundle.mean_edge_time,
        "edge_attributes": {
            key: {
                "min": min(vals),
                "median": statistics.median(vals),
                "mean": statistics.fmean(vals),
                "max": max(vals),
                "nonzero": sum(1 for v in vals if v > 0),
            }
            for key, vals in edge_values.items()
        },
        "rain_activation": {},
    }
    for ri_key in ri_keys:
        network = networks[(ri_key, sampling_profile)]
        blocked_undirected = {canonical_edge(u, v) for u, v in network.blocked_edges}
        components = sorted(
            nx.strongly_connected_components(network.passable_graph),
            key=len,
            reverse=True,
        )
        report["rain_activation"][ri_key] = {
            "activation_model": network.activation_model,
            "soft_planning_full_graph": int(network.plan_with_soft_blocks),
            "hard_blocking_enabled": int(network.hard_blocking_enabled),
            "fail_on_blocked_traversal": int(network.fail_on_blocked_traversal),
            "blocked_directed_edges": len(network.blocked_edges),
            "blocked_undirected_edges": len(blocked_undirected),
            "flood_active_directed_edges": sum(
                1 for _, _, data in network.full_graph.edges(data=True)
                if int(data.get("flood_active", 0)) == 1
            ),
            "landslide_active_directed_edges": sum(
                1 for _, _, data in network.full_graph.edges(data=True)
                if int(data.get("landslide_active", 0)) == 1
            ),
            "flood_blocked_directed_edges": sum(
                1 for _, _, data in network.full_graph.edges(data=True)
                if int(data.get("flood_blocked", 0)) == 1
            ),
            "landslide_blocked_directed_edges": sum(
                1 for _, _, data in network.full_graph.edges(data=True)
                if int(data.get("landslide_blocked", 0)) == 1
            ),
            "largest_passable_component_nodes": len(components[0]) if components else 0,
            "passable_directed_edges": network.passable_graph.number_of_edges(),
        }
    return report


def apply_quick_overrides(cfg: dict[str, Any]) -> None:
    cfg["training"]["scenarios_per_ri"] = 20
    cfg["training"]["epochs"] = 6
    cfg["training"]["early_stop_patience"] = 4
    cfg["evaluation"]["scenarios_per_ri"] = 10
    ddqn_cfg = cfg.setdefault("offline_ddqn", {})
    ddqn_cfg["epochs"] = min(int(ddqn_cfg.get("epochs", 2)), 2)
    ddqn_cfg["batch_size"] = min(int(ddqn_cfg.get("batch_size", 128)), 128)
    online_cfg = cfg.setdefault("online_ddqn", {})
    if bool(online_cfg.get("enabled", False)):
        online_cfg["epochs"] = min(int(online_cfg.get("epochs", 2)), 2)
        online_cfg["batch_size"] = min(int(online_cfg.get("batch_size", 64)), 64)
        online_cfg["min_replay_to_train"] = min(int(online_cfg.get("min_replay_to_train", 128)), 128)
        online_cfg["replay_capacity"] = min(int(online_cfg.get("replay_capacity", 2000)), 2000)
    for cohort in cfg.get("evaluation", {}).get("cohorts", []):
        cohort["scenarios_per_ri"] = min(int(cohort.get("scenarios_per_ri", 10)), 10)


def sampling_mode_for(cfg: dict[str, Any], section: str) -> str:
    return str(
        cfg.get(section, {}).get(
            "sampling_mode",
            cfg.get("scenario", {}).get("sampling_mode", "largest_passable_component"),
        )
    )


def sampling_options_for(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    options.update(cfg.get("scenario", {}).get("sampling_options", {}))
    options.update(cfg.get("scenario", {}).get("hazard_opportunity", {}))
    options.update(cfg.get(section, {}).get("sampling_options", {}))
    options.update(cfg.get(section, {}).get("hazard_opportunity", {}))
    return options


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean macro-DQN thesis framework.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["all", "report", "train", "eval"], default="all")
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run.")
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--ri-keys", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.quick:
        apply_quick_overrides(cfg)

    artifact_dir = resolve_path(cfg["outputs"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    random.seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))
    torch.manual_seed(int(cfg.get("seed", 0)))

    bundle = load_hazard_graph(cfg)
    rains = parse_rain_levels(cfg)
    profiles = parse_profiles(cfg)
    ri_keys = args.ri_keys or list(cfg["scenario"]["ri_keys"])
    profile_keys = args.profiles or list(profiles.keys())
    sampling_profile = "balanced" if "balanced" in profile_keys else profile_keys[0]
    networks = make_networks(bundle, rains, profiles, cfg, ri_keys, profile_keys)

    if args.mode in {"all", "report"}:
        report = graph_report(bundle, networks, ri_keys, sampling_profile)
        with (artifact_dir / "graph_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"Wrote graph report: {artifact_dir / 'graph_report.json'}")

    train_scenarios: list[Scenario] = []
    if args.mode in {"all", "train"}:
        train_scenarios = generate_scenarios(
            networks=networks,
            ri_keys=ri_keys,
            profile_for_sampling=sampling_profile,
            scenarios_per_ri=int(cfg["training"]["scenarios_per_ri"]),
            num_deliveries=int(cfg["scenario"]["num_deliveries"]),
            max_steps=int(cfg["scenario"]["max_steps"]),
            seed=int(cfg.get("seed", 0)),
            scenario_prefix="train",
            sampling_mode=sampling_mode_for(cfg, "training"),
            sampling_options=sampling_options_for(cfg, "training"),
        )
        write_scenarios(artifact_dir / "training_scenarios.jsonl", train_scenarios)
        summaries = []
        for profile_key in profile_keys:
            print(f"Training profile: {profile_key}")
            summaries.append(
                train_profile_model(bundle, networks, train_scenarios, profile_key, cfg, artifact_dir)
            )
        write_csv(artifact_dir / "training_summary.csv", summaries)
        print(f"Wrote training summary: {artifact_dir / 'training_summary.csv'}")

    if args.mode in {"all", "eval"}:
        def run_eval_cohort(
            cohort_name: str,
            eval_cfg: dict[str, Any],
            results_dir_name: str,
            scenario_path_name: str,
        ) -> None:
            options = sampling_options_for(cfg, "evaluation")
            options.update(eval_cfg.get("sampling_options", {}))
            options.update(eval_cfg.get("hazard_opportunity", {}))
            scenario_source = eval_cfg.get("scenario_source_path")
            if scenario_source:
                eval_scenarios = read_scenarios(resolve_path(scenario_source))
                wanted_ri = set(eval_cfg.get("ri_keys", ri_keys))
                eval_scenarios = [
                    scenario for scenario in eval_scenarios
                    if scenario.ri_key in wanted_ri
                ]
                per_ri_limit = int(eval_cfg["scenarios_per_ri"])
                kept_counts: dict[str, int] = defaultdict(int)
                limited_scenarios: list[Scenario] = []
                for scenario in eval_scenarios:
                    if kept_counts[scenario.ri_key] >= per_ri_limit:
                        continue
                    limited_scenarios.append(scenario)
                    kept_counts[scenario.ri_key] += 1
                eval_scenarios = limited_scenarios
            else:
                eval_scenarios = generate_scenarios(
                    networks=networks,
                    ri_keys=list(eval_cfg.get("ri_keys", ri_keys)),
                    profile_for_sampling=sampling_profile,
                    scenarios_per_ri=int(eval_cfg["scenarios_per_ri"]),
                    num_deliveries=int(cfg["scenario"]["num_deliveries"]),
                    max_steps=int(cfg["scenario"]["max_steps"]),
                    seed=int(eval_cfg.get("seed", cfg["evaluation"].get("seed", 4040))),
                    scenario_prefix=f"eval_{cohort_name}",
                    sampling_mode=str(eval_cfg.get("sampling_mode", sampling_mode_for(cfg, "evaluation"))),
                    sampling_options=options,
                )
            write_scenarios(artifact_dir / scenario_path_name, eval_scenarios)
            results = evaluate(
                bundle, networks, eval_scenarios, profile_keys, cfg, artifact_dir, eval_cfg
            )
            raw_rows = [result.to_row() for result in results]
            summary_rows = summarize_results(results)
            results_dir = artifact_dir / results_dir_name
            write_csv(results_dir / "evaluation_raw.csv", raw_rows)
            write_csv(results_dir / "evaluation_summary.csv", summary_rows)
            write_deltas(summary_rows, results_dir / "baseline_deltas.csv")
            write_profile_specialization(
                summary_rows, results_dir / "profile_specialization_summary.csv"
            )
            print(f"Wrote {cohort_name} evaluation raw: {results_dir / 'evaluation_raw.csv'}")
            print(f"Wrote {cohort_name} evaluation summary: {results_dir / 'evaluation_summary.csv'}")
            print(f"Wrote {cohort_name} baseline deltas: {results_dir / 'baseline_deltas.csv'}")
            print(
                f"Wrote {cohort_name} profile specialization: "
                f"{results_dir / 'profile_specialization_summary.csv'}"
            )

        if not bool(cfg["evaluation"].get("skip_main", False)):
            run_eval_cohort(
                cohort_name="main",
                eval_cfg=cfg["evaluation"],
                results_dir_name="results",
                scenario_path_name="evaluation_scenarios.jsonl",
            )
        for cohort in cfg["evaluation"].get("cohorts", []):
            cohort_name = str(cohort["name"])
            cohort_cfg = dict(cfg["evaluation"])
            cohort_cfg.update(cohort)
            run_eval_cohort(
                cohort_name=cohort_name,
                eval_cfg=cohort_cfg,
                results_dir_name=f"results_{cohort_name}",
                scenario_path_name=f"evaluation_scenarios_{cohort_name}.jsonl",
            )

    return 0


def write_scenarios(path: Path, scenarios: list[Scenario]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for scenario in scenarios:
            f.write(
                json.dumps(
                    {
                        "scenario_id": scenario.scenario_id,
                        "ri_key": scenario.ri_key,
                        "start": scenario.start,
                        "deliveries": list(scenario.deliveries),
                        "max_steps": scenario.max_steps,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            f.write("\n")


def read_scenarios(path: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scenarios.append(
                Scenario(
                    scenario_id=str(row["scenario_id"]),
                    ri_key=str(row["ri_key"]),
                    start=str(row["start"]),
                    deliveries=tuple(str(node) for node in row["deliveries"]),
                    max_steps=int(row["max_steps"]),
                )
            )
    return scenarios


if __name__ == "__main__":
    raise SystemExit(main())
