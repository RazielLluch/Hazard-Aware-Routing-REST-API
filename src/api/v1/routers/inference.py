import asyncio

from fastapi import APIRouter

from ....schemas.inference import InferenceHealth, InferenceRequest, InferenceResponse
from ....services import inference_service

DEFAULT_GRAPH_ID = "la_trinidad_subgraph_n200"

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest) -> InferenceResponse:
    return await asyncio.to_thread(inference_service.route, DEFAULT_GRAPH_ID, request)


@router.get("/health", response_model=InferenceHealth)
def health() -> InferenceHealth:
    return inference_service.health()
