"""
GET /api/metrics/pipeline  — real Kafka lag, stream processor stats, forecast freshness
GET /api/metrics/demand    — city-wide surge and demand summary
GET /api/metrics/drift     — PSI scores vs 7-day baseline
GET /api/health            — all service statuses
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas import (
    DemandMetrics, DriftFeature, DriftMetrics, HealthResponse, PipelineMetrics,
)
from src.cache import get_redis
from src.config import get_settings
from src.db import get_pool

log      = logging.getLogger(__name__)
router   = APIRouter(prefix="/api", tags=["metrics"])
settings = get_settings()

_executor = ThreadPoolExecutor(max_workers=2)


def _kafka_lag_sync() -> int:
    """Compute consumer group lag for stream-processor-group on raw_trips topic."""
    from kafka import KafkaAdminClient, KafkaConsumer

    try:
        admin = KafkaAdminClient(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            request_timeout_ms=5_000,
            client_id="backend-health",
        )
        offsets = admin.list_consumer_group_offsets(settings.KAFKA_CONSUMER_GROUP_STREAM)
        admin.close()

        if not offsets:
            return 0

        consumer = KafkaConsumer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            request_timeout_ms=5_000,
        )
        end_offsets = consumer.end_offsets(list(offsets.keys()))
        consumer.close()

        lag = sum(
            max(0, end_offsets.get(tp, 0) - meta.offset)
            for tp, meta in offsets.items()
        )
        return int(lag)
    except Exception as exc:
        log.warning("Kafka lag query failed: %s", exc)
        return -1


async def _get_kafka_lag() -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _kafka_lag_sync)


@router.get("/metrics/pipeline", response_model=PipelineMetrics)
async def pipeline_metrics():
    redis = get_redis()

    kafka_lag, stats, redis_writes = await asyncio.gather(
        _get_kafka_lag(),
        redis.hgetall("processor:stats"),
        redis.get("processor:redis_ops"),
    )

    now = datetime.utcnow()

    # Seconds since last window flush
    last_window_str = stats.get("last_window_time") if stats else None
    if last_window_str:
        try:
            last_window = datetime.fromisoformat(last_window_str)
            seconds_since_window = (now - last_window).total_seconds()
        except ValueError:
            seconds_since_window = -1.0
    else:
        seconds_since_window = -1.0

    # Seconds since last forecast
    pool = get_pool()
    async with pool.acquire() as conn:
        last_fc_row = await conn.fetchrow(
            "SELECT MAX(created_at) AS last_fc FROM forecasts"
        )
    last_fc = last_fc_row["last_fc"] if last_fc_row else None
    if last_fc:
        last_fc_naive = last_fc.replace(tzinfo=None) if last_fc.tzinfo else last_fc
        seconds_since_forecast = (now - last_fc_naive).total_seconds()
    else:
        seconds_since_forecast = -1.0

    processor_status = "healthy"
    if seconds_since_window < 0 or seconds_since_window > 300:
        processor_status = "stale"

    return PipelineMetrics(
        kafka_consumer_lag          = kafka_lag,
        seconds_since_last_window   = round(seconds_since_window, 1),
        redis_writes_total          = int(redis_writes or 0),
        seconds_since_last_forecast = round(seconds_since_forecast, 1),
        total_events_processed      = int(stats.get("total_events", 0) if stats else 0),
        active_zones_last_window    = int(stats.get("last_zones_count", 0) if stats else 0),
        stream_processor_status     = processor_status,
    )


@router.get("/metrics/demand", response_model=DemandMetrics)
async def demand_metrics():
    redis = get_redis()
    pool  = get_pool()

    # Load all zone_state keys from Redis
    zone_keys = await redis.keys("zone_state:*")
    if not zone_keys:
        return DemandMetrics(
            total_active_zones  = 0,
            zones_in_surge      = 0,
            avg_surge_signal    = 1.0,
            max_surge_signal    = 1.0,
            highest_demand_zone = None,
            demand_vs_baseline  = 0.0,
        )

    pipe = redis.pipeline()
    for key in zone_keys:
        pipe.hgetall(key)
    states = await pipe.execute()

    surge_signals    = []
    demands          = []
    highest_demand   = 0
    highest_demand_h3 = None

    for key, state in zip(zone_keys, states):
        if not state:
            continue
        surge  = float(state.get("surge_signal", 1.0))
        demand = int(state.get("current_demand", 0))
        surge_signals.append(surge)
        demands.append(demand)
        if demand > highest_demand:
            highest_demand     = demand
            highest_demand_h3  = key.replace("zone_state:", "")

    zones_in_surge = sum(1 for s in surge_signals if s > 1.2)
    avg_surge      = sum(surge_signals) / len(surge_signals) if surge_signals else 1.0
    max_surge      = max(surge_signals, default=1.0)

    # Demand vs 7-day baseline: compare current avg demand to historical avg
    async with pool.acquire() as conn:
        baseline_row = await conn.fetchrow(
            """
            SELECT AVG(trip_count) AS baseline
            FROM zone_features
            WHERE pickup_hour = EXTRACT(HOUR FROM NOW())::INT
              AND window_start >= NOW() - INTERVAL '7 days'
            """
        )
    baseline = float(baseline_row["baseline"] or 1.0) if baseline_row else 1.0
    current_avg = sum(demands) / len(demands) if demands else 0.0
    demand_vs_baseline = round(((current_avg - baseline) / baseline) * 100, 1) if baseline > 0 else 0.0

    return DemandMetrics(
        total_active_zones  = len(surge_signals),
        zones_in_surge      = zones_in_surge,
        avg_surge_signal    = round(avg_surge, 3),
        max_surge_signal    = round(max_surge, 3),
        highest_demand_zone = highest_demand_h3,
        demand_vs_baseline  = demand_vs_baseline,
    )


@router.get("/metrics/drift", response_model=DriftMetrics)
async def drift_metrics():
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (feature_name)
                   feature_name, psi_score, baseline_mean, current_mean,
                   alert_triggered, computed_at
            FROM drift_metrics
            ORDER BY feature_name, computed_at DESC
            """
        )

    if not rows:
        return DriftMetrics(computed_at=datetime.utcnow().isoformat(), features=[])

    features = [
        DriftFeature(
            feature_name    = r["feature_name"],
            psi_score       = float(r["psi_score"] or 0),
            baseline_mean   = float(r["baseline_mean"] or 0),
            current_mean    = float(r["current_mean"] or 0),
            alert_triggered = bool(r["alert_triggered"]),
        )
        for r in rows
    ]
    latest_time = max(r["computed_at"] for r in rows)

    return DriftMetrics(
        computed_at = latest_time.isoformat(),
        features    = features,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    pool  = get_pool()
    redis = get_redis()

    # PostgreSQL
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        pg_status = "healthy"
    except Exception:
        pg_status = "unhealthy"

    # Redis
    try:
        await redis.ping()
        redis_status = "healthy"
    except Exception:
        redis_status = "unhealthy"

    # Kafka (sync, in executor)
    loop = asyncio.get_event_loop()
    def _kafka_check():
        from kafka import KafkaConsumer
        try:
            c = KafkaConsumer(bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                              request_timeout_ms=3_000)
            c.topics()
            c.close()
            return "healthy"
        except Exception:
            return "unhealthy"

    kafka_status = await loop.run_in_executor(_executor, _kafka_check)

    overall = "healthy" if all(s == "healthy" for s in [pg_status, redis_status, kafka_status]) else "degraded"

    return HealthResponse(
        status    = overall,
        postgres  = pg_status,
        redis     = redis_status,
        kafka     = kafka_status,
        timestamp = datetime.utcnow().isoformat(),
    )
