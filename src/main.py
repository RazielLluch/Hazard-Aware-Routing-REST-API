from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v2 import router as v2_router
from src.core.config import settings
from src.core.errors import register_exception_handlers
from src.core.logging import configure_logging, get_logger
from src.macro.bridge import bridge


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log = get_logger("startup")
    log.info(
        "starting",
        data_root=str(settings.data_root),
        macro_vendor_root=str(settings.macro_vendor_root),
    )
    # Fail fast at boot if the vendored Macro-DDQN runner cannot be loaded.
    bridge.load()
    log.info("macro_bridge_ready", feature_count=len(bridge.feature_names()))
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

app.include_router(v2_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}
