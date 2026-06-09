"""PulseMap FastAPI backend."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import events, metrics, zones
from src.cache import close_redis, init_redis
from src.config import get_settings
from src.db import close_pool, init_pool

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("PulseMap backend starting")
    await init_pool(settings.DATABASE_URL)
    init_redis(settings.REDIS_URL)
    log.info("Database pool and Redis ready")
    yield
    await close_pool()
    await close_redis()
    log.info("PulseMap backend shut down")


app = FastAPI(
    title       = "PulseMap",
    description = "Real-Time Ride Demand Intelligence Platform",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(zones.router)
app.include_router(events.router)
app.include_router(metrics.router)
