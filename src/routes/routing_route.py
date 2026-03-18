import uuid

from src.models.route_model import RouteResponseModel, RouteRequestModel
from fastapi import APIRouter, Body, Depends, HTTPException, status, Path, Response
from ..services.mock_data import generate_mock_routes

api_router = APIRouter()


@api_router.get("/")
def hello_world():
    return {"message": "Hello World"}


@api_router.post(
    "/route/generate_mock/",
    response_model=RouteResponseModel,
    status_code=status.HTTP_201_CREATED,
    summary="Creates a new route from input nodes",
    tags=["Routes"],
)
async def generate_route(
        request: RouteRequestModel
):

    response = generate_mock_routes(request)

    return response
