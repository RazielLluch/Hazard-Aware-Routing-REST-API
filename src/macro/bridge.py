"""FastAPI bridge into the vendored Macro-DDQN runner.

This module is the single seam between the FastAPI service and the vendored
``runner.py`` (a verbatim copy of ``macro_DDQN/run_macro_dqn_final.py``). It
mirrors the proven pattern in ``macro_DDQN/api/live_route_api.py`` — reusing the
runner's primitives (``load_hazard_graph``, ``make_networks``, ``feature_vector``,
``predict_cost``, ``optimal_order``, ``add_metrics`` ...) and reimplementing the
macro policy loop — but adds module-scope caching: the graph ``bundle`` is loaded
once, and each ``(ri, profile)`` ``ActivatedNetwork`` plus each model checkpoint
is built lazily and memoised.

Algorithm identity: the runner collapses every learned variant to the single
label ``Macro-DQN``. The 7-entry catalog below is the only place the variants
become distinct, keyed by artefact directory — see ``_ALGORITHM_SPECS``.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger("macro.bridge")


# ---------------------------------------------------------------------------
# Algorithm catalog — the 7 thesis algorithms, keyed by artefact directory.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlgorithmSpec:
    """One entry in the comparison surface.

    ``runner_algorithm`` is the label the policy loop branches on (the runner
    only knows ``Macro-DQN`` / ``Macro-DP-Oracle`` / ``Greedy-*``).
    ``artefact_dir`` is the sub-directory under ``artifacts/`` that holds the
    checkpoint for learned variants (``None`` for oracle/greedy).
    """

    id: str
    label: str
    category: str  # "learned" | "baseline" | "oracle"
    requires_model: bool
    runner_algorithm: str
    artefact_dir: str | None
    aliases: tuple[str, ...]


_ALGORITHM_SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        id="Macro-DQN",
        label="Macro-DQN (DP-pretrained reference)",
        category="learned",
        requires_model=True,
        runner_algorithm="Macro-DQN",
        artefact_dir="final_macro_dqn_v2_25yr_adjusted_s500_e300_with_cohorts",
        aliases=("macro-dqn", "macro_dqn", "dqn", "pretrained", "reference"),
    ),
    AlgorithmSpec(
        id="Macro-DDQN-Online-Scratch",
        label="Macro-DDQN (online, scratch)",
        category="learned",
        requires_model=True,
        runner_algorithm="Macro-DQN",
        artefact_dir="final_macro_dqn_v2_25yr_adjusted_s500_e300_with_cohorts_online_ddqn_scratch",
        aliases=("macro-ddqn-online-scratch", "online-scratch", "ddqn-scratch", "scratch"),
    ),
    AlgorithmSpec(
        id="Macro-DDQN-Offline-NoAnchor",
        label="Macro-DDQN (offline, no anchor)",
        category="learned",
        requires_model=True,
        runner_algorithm="Macro-DQN",
        artefact_dir="final_macro_dqn_v2_25yr_adjusted_s500_e300_with_cohorts_offline_ddqn_no_anchor",
        aliases=("macro-ddqn-offline-no-anchor", "offline-no-anchor", "ddqn-offline", "no-anchor"),
    ),
    AlgorithmSpec(
        id="Macro-DP-Oracle",
        label="Macro-DP Oracle (exact)",
        category="oracle",
        requires_model=False,
        runner_algorithm="Macro-DP-Oracle",
        artefact_dir=None,
        aliases=("macro-dp-oracle", "macro_dp_oracle", "oracle", "dp"),
    ),
    AlgorithmSpec(
        id="Greedy-Time-Base",
        label="Greedy (time, hazard-blind)",
        category="baseline",
        requires_model=False,
        runner_algorithm="Greedy-Time-Base",
        artefact_dir=None,
        aliases=("greedy-time-base", "greedy_time_base", "time-base"),
    ),
    AlgorithmSpec(
        id="Greedy-Time-SoftPenalty",
        label="Greedy (time, soft-penalty)",
        category="baseline",
        requires_model=False,
        runner_algorithm="Greedy-Time-SoftPenalty",
        artefact_dir=None,
        aliases=("greedy-time-softpenalty", "greedy_time_softpenalty", "greedy-softpenalty"),
    ),
    AlgorithmSpec(
        id="Greedy-HazardAware",
        label="Greedy (hazard-aware)",
        category="baseline",
        requires_model=False,
        runner_algorithm="Greedy-HazardAware",
        artefact_dir=None,
        aliases=("greedy-hazardaware", "greedy_hazardaware", "hazardaware"),
    ),
)

_SPEC_BY_KEY: dict[str, AlgorithmSpec] = {}
for _spec in _ALGORITHM_SPECS:
    _SPEC_BY_KEY[_spec.id.lower()] = _spec
    for _alias in _spec.aliases:
        _SPEC_BY_KEY[_alias.lower()] = _spec


def resolve_algorithm(value: str) -> AlgorithmSpec:
    """Map any of the 3 spellings (id / kebab alias / runner column) to a spec."""
    spec = _SPEC_BY_KEY.get(str(value).strip().lower())
    if spec is None:
        known = ", ".join(s.id for s in _ALGORITHM_SPECS)
        raise ValueError(f"Unknown algorithm {value!r}. Known: {known}.")
    return spec


# ---------------------------------------------------------------------------
# Payload helpers — ported verbatim from macro_DDQN/api/live_route_api.py so the
# /api/v2/inference response matches the proven route_response() shape.
# ---------------------------------------------------------------------------
def _fnum(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _node_payload(bundle: Any, node_id: str) -> dict[str, Any]:
    data = bundle.graph.nodes[str(node_id)]
    x = _fnum(data.get("x", 0.0))
    y = _fnum(data.get("y", 0.0))
    return {"id": str(node_id), "x": x, "y": y, "lng": x, "lat": y}


def _edge_payload(bundle: Any, network: Any, u: str, v: str) -> dict[str, Any]:
    data = network.full_graph[str(u)][str(v)]
    flood_exposure = _fnum(data.get("flood_exposure", 0.0))
    landslide_exposure = _fnum(data.get("landslide_exposure", 0.0))
    blockage_exposure = _fnum(data.get("blockage_exposure", 0.0))
    common_risk = (
        network.common_hazard_weight * (flood_exposure + landslide_exposure)
        + blockage_exposure
    )
    return {
        "u": str(u),
        "v": str(v),
        "coordinates": [_node_payload(bundle, str(u)), _node_payload(bundle, str(v))],
        "length_m": _fnum(data.get("length_m", data.get("length", 0.0))),
        "base_time_min": _fnum(data.get("base_time", 0.0)),
        "travel_time_min": _fnum(data.get("travel_time", 0.0)),
        "objective": _fnum(data.get("objective", 0.0)),
        "flood_hazard": _fnum(data.get("flood_hazard", data.get("flood_score", 0.0))),
        "landslide_hazard": _fnum(data.get("landslide_hazard", data.get("landslide_score", 0.0))),
        "flood_active": int(data.get("flood_active", 0)),
        "landslide_active": int(data.get("landslide_active", 0)),
        "flood_blocked": int(data.get("flood_blocked", 0)),
        "landslide_blocked": int(data.get("landslide_blocked", 0)),
        "blocked": int(data.get("blocked", 0)),
        "flood_exposure": flood_exposure,
        "landslide_exposure": landslide_exposure,
        "blockage_exposure": blockage_exposure,
        "risk_exposure": _fnum(data.get("risk", 0.0)),
        "common_risk_exposure": _fnum(common_risk),
    }


def _metrics_payload(metrics: Any, common_hazard_weight: float) -> dict[str, Any]:
    return {
        "objective": _fnum(metrics.objective),
        "time_min": _fnum(metrics.travel_time_min),
        "distance_m": _fnum(metrics.distance_m),
        "distance_km": _fnum(metrics.distance_m / 1000.0),
        "flood_exposure": _fnum(metrics.flood_exposure),
        "landslide_exposure": _fnum(metrics.landslide_exposure),
        "blockage_exposure": _fnum(metrics.blockage_exposure),
        "risk_exposure": _fnum(metrics.risk_exposure),
        "common_risk_exposure": _fnum(
            common_hazard_weight * (metrics.flood_exposure + metrics.landslide_exposure)
            + metrics.blockage_exposure
        ),
        "hops": int(metrics.hops),
    }


def _leg_payload(bundle: Any, network: Any, source: str, target: str, metrics: Any) -> dict[str, Any]:
    path = [str(node) for node in metrics.path]
    return {
        "source": str(source),
        "target": str(target),
        "path_nodes": path,
        "metrics": _metrics_payload(metrics, network.common_hazard_weight),
        "segments": [_edge_payload(bundle, network, u, v) for u, v in zip(path, path[1:])],
    }


def _model_choice(
    runner: Any,
    bundle: Any,
    network: Any,
    current: str,
    remaining: list[str],
    num_deliveries: int,
    model: Any,
    checkpoint: dict[str, Any],
) -> tuple[str | None, Any | None]:
    """Argmin of the Q-network's predicted cost-to-go over remaining candidates."""
    features = [
        runner.feature_vector(bundle, network, current, target, tuple(remaining), num_deliveries)
        for target in remaining
    ]
    predicted_costs = runner.predict_cost(model, checkpoint, features)
    ranked = sorted(zip(predicted_costs, remaining), key=lambda item: item[0])
    for _, target in ranked:
        path = network.path(current, target, weight="objective", passable=True)
        if path is not None:
            return target, path
    return None, None


def _greedy_choice(
    network: Any, current: str, remaining: list[str], runner_algorithm: str
) -> tuple[str | None, Any | None]:
    """Locally-greedy next-delivery pick under the algorithm's Dijkstra weight."""
    best: tuple[float, str, Any] | None = None
    for target in remaining:
        if runner_algorithm == "Greedy-Time-Base":
            plan = network.path(current, target, weight="base_time", passable=False)
            if plan is None:
                continue
            cost = sum(
                float(network.full_graph[u][v]["base_time"])
                for u, v in zip(plan.path, plan.path[1:])
            )
            path = network.execute_planned_path(plan.path, passable=False, weight="base_time")
        elif runner_algorithm == "Greedy-Time-SoftPenalty":
            plan = network.path(current, target, weight="travel_time", passable=False)
            if plan is None:
                continue
            cost = plan.travel_time_min
            path = network.execute_planned_path(plan.path, passable=False, weight="travel_time")
        else:  # Greedy-HazardAware
            path = network.path(current, target, weight="objective", passable=True)
            cost = path.objective if path is not None else math.inf
        if path is None:
            continue
        if best is None or cost < best[0]:
            best = (cost, target, path)
    if best is None:
        return None, None
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Bridge — module-scope singleton, loaded once at FastAPI lifespan startup.
# ---------------------------------------------------------------------------
class Bridge:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runner: Any | None = None
        self._cfg: dict[str, Any] | None = None
        self._bundle: Any | None = None
        self._config_path: Path | None = None
        self._networks: dict[tuple[str, str], Any] = {}
        self._models: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}

    # ---- lifecycle --------------------------------------------------------
    def load(self, config_path: Path | str | None = None) -> None:
        """Load the vendored runner, parse the config, and build the graph bundle.

        Idempotent-friendly: safe to call again to switch configs. The heavy
        ``runner`` import (torch / networkx) is deferred to here so importing
        this module stays cheap.
        """
        with self._lock:
            os.environ.setdefault("MACRO_DDQN_ROOT", str(settings.macro_vendor_root))
            from src.macro import runner  # noqa: PLC0415 - deferred heavy import

            path = Path(config_path) if config_path else self._default_config_path()
            if not path.exists():
                raise FileNotFoundError(f"Macro config not found: {path}")
            self._runner = runner
            self._cfg = runner.load_config(path)
            self._config_path = path
            self._bundle = runner.load_hazard_graph(self._cfg)
            self._networks.clear()
            self._models.clear()
            log.info(
                "macro_bridge_loaded",
                config=str(path),
                repo_root=str(runner.REPO_ROOT),
                feature_count=len(runner.FEATURE_NAMES),
                nodes=self._bundle.graph.number_of_nodes(),
            )

    def reload(self) -> None:
        """Drop all caches and reload from the current config path."""
        with self._lock:
            cfg_path = self._config_path
            self._runner = None
            self._cfg = None
            self._bundle = None
            self._networks.clear()
            self._models.clear()
        self.load(cfg_path)

    @property
    def is_loaded(self) -> bool:
        return self._runner is not None

    @property
    def runner(self) -> Any:
        if self._runner is None:
            raise RuntimeError("Bridge not loaded — call bridge.load() first.")
        return self._runner

    @property
    def cfg(self) -> dict[str, Any]:
        if self._cfg is None:
            raise RuntimeError("Bridge not loaded — call bridge.load() first.")
        return self._cfg

    @property
    def bundle(self) -> Any:
        if self._bundle is None:
            raise RuntimeError("Bridge not loaded — call bridge.load() first.")
        return self._bundle

    # ---- path resolution --------------------------------------------------
    def _macro_root(self) -> Path:
        return Path(settings.macro_vendor_root) / "macro_DDQN"

    def _default_config_path(self) -> Path:
        return self._macro_root() / "configs" / settings.macro_default_config

    def _artefact_root(self) -> Path:
        return self._macro_root() / "artifacts"

    # ---- cached builders --------------------------------------------------
    def network(self, ri: str, profile: str) -> Any:
        """Lazily build and memoise the ``ActivatedNetwork`` for ``(ri, profile)``."""
        key = (ri, profile)
        with self._lock:
            cached = self._networks.get(key)
            if cached is not None:
                return cached
            runner = self.runner
            rains = runner.parse_rain_levels(self.cfg)
            profiles = runner.parse_profiles(self.cfg)
            if ri not in rains:
                raise ValueError(f"Unknown RI key: {ri!r}")
            if profile not in profiles:
                raise ValueError(f"Unknown profile: {profile!r}")
            networks = runner.make_networks(
                bundle=self.bundle,
                rains=rains,
                profiles=profiles,
                cfg=self.cfg,
                ri_keys=[ri],
                profile_keys=[profile],
            )
            net = networks[(ri, profile)]
            self._networks[key] = net
            return net

    def _model(self, profile: str, spec: AlgorithmSpec) -> tuple[Any, dict[str, Any]]:
        """Lazily load and memoise the checkpoint for a learned algorithm."""
        if spec.artefact_dir is None:
            raise ValueError(f"{spec.id} does not use a model checkpoint.")
        key = (spec.id, profile)
        with self._lock:
            cached = self._models.get(key)
            if cached is not None:
                return cached
            model_path = (
                self._artefact_root() / spec.artefact_dir / "models" / f"macro_dqn_{profile}.pt"
            )
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found for {spec.id} / {profile}: {model_path}"
                )
            loaded = self.runner.load_model(model_path)
            self._models[key] = loaded
            return loaded

    # ---- public API -------------------------------------------------------
    def feature_names(self) -> list[str]:
        return list(self.runner.FEATURE_NAMES)

    def algorithm_catalog(self) -> list[dict[str, Any]]:
        """The 7-entry catalog backing ``GET /api/v2/algorithms``."""
        return [
            {
                "id": s.id,
                "label": s.label,
                "category": s.category,
                "requires_model": s.requires_model,
                "aliases": list(s.aliases),
            }
            for s in _ALGORITHM_SPECS
        ]

    def graph_export(self) -> dict[str, Any]:
        """Full graph + all (RI, profile) networks — the frontend map backdrop.

        Mirrors ``export_viewer_data.export_graph()`` but reuses the cached
        ``bundle`` and memoised ``network()`` instances. Expensive on a cold
        cache (builds every (RI, profile) ``ActivatedNetwork``); callers should
        cache the result — the graph is invariant within a config.
        """
        cfg = self.cfg
        bundle = self.bundle
        ri_keys = list(cfg["scenario"]["ri_keys"])
        profile_keys = list(self.runner.parse_profiles(cfg).keys())
        nodes = [
            {"id": str(nid), "x": _fnum(data.get("x", 0.0)), "y": _fnum(data.get("y", 0.0))}
            for nid, data in bundle.graph.nodes(data=True)
        ]
        networks_out: dict[str, list[dict[str, Any]]] = {}
        for ri in ri_keys:
            for profile in profile_keys:
                net = self.network(ri, profile)
                edges: list[dict[str, Any]] = []
                for u, v, data in net.full_graph.edges(data=True):
                    flood = _fnum(data.get("flood_hazard", 0.0))
                    landslide = _fnum(data.get("landslide_hazard", 0.0))
                    edges.append(
                        {
                            "u": str(u),
                            "v": str(v),
                            "base_time": _fnum(data.get("base_time", 0.0)),
                            "travel_time": _fnum(data.get("travel_time", 0.0)),
                            "objective": _fnum(data.get("objective", 0.0)),
                            "length_m": _fnum(data.get("length_m", 0.0)),
                            "flood": flood,
                            "landslide": landslide,
                            "combined": _fnum(data.get("combined_hazard", max(flood, landslide))),
                            "flood_active": int(data.get("flood_active", 0)),
                            "landslide_active": int(data.get("landslide_active", 0)),
                            "flood_blocked": int(data.get("flood_blocked", 0)),
                            "landslide_blocked": int(data.get("landslide_blocked", 0)),
                            "blocked": int(data.get("blocked", 0)),
                        }
                    )
                networks_out[f"{ri}:{profile}"] = edges
        return {
            "nodes": nodes,
            "networks": networks_out,
            "profiles": profile_keys,
            "ri_keys": ri_keys,
            "algorithms": [entry["id"] for entry in self.algorithm_catalog()],
            "artefact_id": cfg.get("run_name"),
        }

    def optimal_order(
        self, ri: str, profile: str, start: str, deliveries: list[str]
    ) -> tuple[float, tuple[str, ...]]:
        net = self.network(ri, profile)
        return self.runner.optimal_order(net, start, tuple(deliveries))

    def compute_route(
        self,
        *,
        algorithm: str,
        profile: str,
        ri: str,
        start: str,
        deliveries: list[str],
        max_steps: int = 1000,
    ) -> dict[str, Any]:
        """Run the macro policy loop for one scenario.

        Returns a ``route_response()``-shaped dict (legs + per-segment payload).
        ``success`` is intentionally omitted — under soft-blocking it is always
        true; callers read ``failure_reason`` + ``delivered`` instead.
        """
        spec = resolve_algorithm(algorithm)
        runner = self.runner
        if not start:
            raise ValueError("`start` is required.")
        if not deliveries:
            raise ValueError("`deliveries` is required.")
        num_deliveries = int(self.cfg["scenario"]["num_deliveries"])
        if len(deliveries) != num_deliveries:
            raise ValueError(
                f"Config expects {num_deliveries} deliveries, got {len(deliveries)}."
            )
        net = self.network(ri, profile)

        model = checkpoint = None
        if spec.requires_model:
            model, checkpoint = self._model(profile, spec)

        ordered_targets: list[str] = []
        if spec.runner_algorithm == "Macro-DP-Oracle":
            _, order = runner.optimal_order(net, start, tuple(deliveries))
            ordered_targets = list(order)

        current = start
        remaining = list(deliveries)
        total = runner.PathMetrics.empty(current)
        legs: list[dict[str, Any]] = []
        visit_order: list[str] = []
        failure_reason = ""

        while remaining:
            if spec.requires_model:
                target, path = _model_choice(
                    runner, self.bundle, net, current, remaining, num_deliveries, model, checkpoint
                )
            elif spec.runner_algorithm == "Macro-DP-Oracle":
                target = ordered_targets.pop(0) if ordered_targets else None
                path = (
                    net.path(current, target, weight="objective", passable=True)
                    if target
                    else None
                )
            else:
                target, path = _greedy_choice(net, current, remaining, spec.runner_algorithm)

            if target is None or path is None:
                failure_reason = "no_route"
                break
            total = runner.add_metrics(total, path)
            legs.append(_leg_payload(self.bundle, net, current, target, path))
            visit_order.append(target)
            remaining.remove(target)
            current = target
            if total.hops > max_steps:
                failure_reason = "timeout"
                break

        return {
            "algorithm": spec.id,
            "profile": profile,
            "ri_key": ri,
            "artefact_id": spec.artefact_dir,
            "delivered": len(visit_order),
            "failure_reason": failure_reason,
            "start": _node_payload(self.bundle, start),
            "deliveries": [_node_payload(self.bundle, node) for node in deliveries],
            "visit_order": visit_order,
            "metrics": _metrics_payload(total, net.common_hazard_weight),
            "legs": legs,
        }


bridge = Bridge()
