import asyncio
import logging
import sys
from contextlib import asynccontextmanager

# psycopg async on Windows requires SelectorEventLoop (not ProactorEventLoop)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import anomalies, baselines, ml_routes, processes, schema_routes, upload
from app.config import settings
from app.database import dispose_engine
from app.services import ml_service

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Logistics Bottleneck Detection API")
    ml_service.init_models(settings.ml_model_dir)
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
app.include_router(ml_routes.router, prefix=API_PREFIX, tags=["ML"])


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    return {"status": "ok", "version": app.version}
