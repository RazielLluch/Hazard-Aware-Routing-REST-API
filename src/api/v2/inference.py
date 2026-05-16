"""POST /api/v2/inference — live Macro-DDQN routing via the vendored runner bridge."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from ...core.errors import InferenceError, ValidationFailure
from ...macro.bridge import bridge
from .schemas import HealthV2, InferenceRequestV2, InferenceResponseV2

router = APIRouter(prefix="/inference", tags=["v2:inference"])


@router.post("", response_model=InferenceResponseV2)
async def run_inference(request: InferenceRequestV2) -> dict:
    """Compute a route for one scenario.

    The macro policy loop (Dijkstra + Q-network forward passes) is CPU-bound, so
    it runs in a worker thread to keep the event loop free.
    """
    try:
        return await asyncio.to_thread(
            bridge.compute_route,
            algorithm=request.algorithm,
            profile=request.profile,
            ri=request.ri_key,
            start=request.start,
            deliveries=request.deliveries,
            max_steps=request.max_steps,
        )
    except ValueError as exc:
        raise ValidationFailure(str(exc)) from exc
    except FileNotFoundError as exc:
        raise InferenceError(str(exc)) from exc


@router.get("/health", response_model=HealthV2)
def health() -> HealthV2:
    """Liveness of the vendored runner bridge."""
    loaded = bridge.is_loaded
    return HealthV2(
        loaded=loaded,
        feature_count=len(bridge.feature_names()) if loaded else 0,
    )
