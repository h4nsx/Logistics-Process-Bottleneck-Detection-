import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import anomalies, baselines, processes, schema_routes, upload
from app.config import settings
from app.database import dispose_engine

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Logistics Bottleneck Detection API")
    yield
    await dispose_engine()
    logger.info("Database engine disposed")


app = FastAPI(
    title="Logistics Process Bottleneck Detection",
    description=(
        "Statistical bottleneck detection for logistics operations. "
        "Detects abnormal step durations using historical baselines."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api"
app.include_router(upload.router, prefix=API_PREFIX, tags=["Upload"])
app.include_router(anomalies.router, prefix=API_PREFIX, tags=["Anomalies"])
app.include_router(baselines.router, prefix=API_PREFIX, tags=["Baselines"])
app.include_router(processes.router, prefix=API_PREFIX, tags=["Processes"])
app.include_router(schema_routes.router, prefix=API_PREFIX, tags=["Schema"])


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    return {"status": "ok", "version": app.version}
