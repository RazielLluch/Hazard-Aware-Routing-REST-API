from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1 import router as v1_router
from src.core.config import settings
from src.core.errors import register_exception_handlers
from src.core.logging import configure_logging, get_logger
from src.routes.routing_route import api_router as legacy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log = get_logger("startup")
    log.info(
        "starting",
        harness_root=str(settings.harness_root),
        graphs_root=str(settings.graphs_root),
        models_root=str(settings.models_root),
    )
    yield
    log.info("shutdown")


app = FastAPI(
    title="Hazard-Aware Routing API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(v1_router)
app.include_router(legacy_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}
