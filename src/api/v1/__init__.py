from fastapi import APIRouter

from .routers import benchmarks, graphs, inference, metrics, runs, scenarios

router = APIRouter(prefix="/api/v1")
router.include_router(benchmarks.router)
router.include_router(scenarios.router)
router.include_router(runs.router)
router.include_router(metrics.router)
router.include_router(graphs.router)
router.include_router(inference.router)
