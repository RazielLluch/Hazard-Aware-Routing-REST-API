"""``/api/v2`` — the sole API surface; Macro-DDQN inference + the 7-algorithm
comparison routers (algorithms, graph, inference, metrics, scenario_sets,
scenarios). Mounted by ``src/main.py``.
"""

from fastapi import APIRouter

from . import algorithms, graph, inference, metrics, scenario_sets, scenarios

router = APIRouter(prefix="/api/v2")
router.include_router(inference.router)
router.include_router(algorithms.router)
router.include_router(graph.router)
router.include_router(metrics.router)
router.include_router(scenario_sets.router)
router.include_router(scenarios.router)
