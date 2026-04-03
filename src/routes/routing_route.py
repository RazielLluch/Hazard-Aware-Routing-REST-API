import uuid

from src.models.route_model import RouteResponseModel, RouteRequestModel
from fastapi import APIRouter, Body, Depends, HTTPException, status, Path, Response

from ..services.inference import inference
from ..services.mock_data import generate_mock_routes
from ..utils.logging_decorator import log_endpoint

api_router = APIRouter()


@api_router.get("/")
def hello_world():
    return {"message": "Hello World"}


@api_router.post(
    "/route/generate_osm/",
    response_model=RouteResponseModel,
    status_code=status.HTTP_201_CREATED,
    summary="Creates a new route from input nodes",
    tags=["Routes"],
)
@log_endpoint
async def generate_osm_route(
        request: RouteRequestModel
):

    response = generate_mock_routes(request)
    return response


@api_router.post(
    "/route/generate/",
    response_model=RouteResponseModel,
    status_code=status.HTTP_201_CREATED,
    summary="Creates a new route from input nodes",
    tags=["Routes"],
)
@log_endpoint
async def generate_route(
        request: RouteRequestModel
):

    response = inference(request)
    response = RouteResponseModel.model_validate(response)
    return response
