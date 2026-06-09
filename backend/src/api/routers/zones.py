"""
GET /api/zones          — all active zones with current state
GET /api/zones/{h3}     — zone detail with 24h history
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.schemas import ZoneDetail, ZoneHistoryPoint, ZoneForecast, ZoneState
from src.cache import get_redis
from src.db import get_pool

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/zones", tags=["zones"])

_EMPTY_STATE = {
    "current_demand":        0,
    "demand_velocity":       0,
    "supply_pressure_index": 1.0,
    "surge_signal":          1.0,
    "window_start":          None,
    "window_end":            None,
}


def _parse_zone_state(meta: dict, state: dict, forecast: dict | None) -> ZoneState:
    fc = None
    if forecast:
        fc = ZoneForecast(
            predicted_demand = float(forecast.get("predicted_demand", 0)),
            confidence_lower = float(forecast.get("confidence_lower", 0)),
            confidence_upper = float(forecast.get("confidence_upper", 0)),
            forecast_time    = forecast.get("forecast_time"),
            model_version    = forecast.get("model_version"),
        )
    return ZoneState(
        h3_index              = meta["h3_index"],
        centroid_lat          = float(meta["centroid_lat"]),
        centroid_lon          = float(meta["centroid_lon"]),
        demand_tier           = meta["demand_tier"],
        current_demand        = int(state.get("current_demand", 0)),
        demand_velocity       = int(state.get("demand_velocity", 0)),
        supply_pressure_index = float(state.get("supply_pressure_index", 1.0)),
        surge_signal          = float(state.get("surge_signal", 1.0)),
        window_start          = state.get("window_start"),
        window_end            = state.get("window_end"),
        forecast              = fc,
    )


@router.get("", response_model=list[ZoneState])
async def list_zones():
    pool  = get_pool()
    redis = get_redis()

    # Load zone metadata from PostgreSQL
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT h3_index, centroid_lat, centroid_lon, demand_tier FROM zone_stats"
        )

    if not rows:
        return []

    # Batch-fetch current state and forecasts from Redis
    pipe = redis.pipeline()
    for row in rows:
        pipe.hgetall(f"zone_state:{row['h3_index']}")
        pipe.hgetall(f"zone_forecast:{row['h3_index']}")
    results = await pipe.execute()

    zones = []
    for i, row in enumerate(rows):
        state    = results[i * 2]
        forecast = results[i * 2 + 1]
        meta     = dict(row)
        zones.append(_parse_zone_state(meta, state or _EMPTY_STATE, forecast or None))

    return zones


@router.get("/{h3_index}", response_model=ZoneDetail)
async def zone_detail(h3_index: str):
    pool  = get_pool()
    redis = get_redis()

    async with pool.acquire() as conn:
        meta_row = await conn.fetchrow(
            "SELECT h3_index, centroid_lat, centroid_lon, demand_tier FROM zone_stats WHERE h3_index = $1",
            h3_index,
        )
        if not meta_row:
            raise HTTPException(status_code=404, detail=f"Zone {h3_index} not found")

        # 24h history from streaming_events
        history_rows = await conn.fetch(
            """
            SELECT window_start, current_demand, surge_signal
            FROM streaming_events
            WHERE h3_index = $1
              AND window_start >= NOW() - INTERVAL '24 hours'
            ORDER BY window_start
            """,
            h3_index,
        )

    pipe = redis.pipeline()
    pipe.hgetall(f"zone_state:{h3_index}")
    pipe.hgetall(f"zone_forecast:{h3_index}")
    state, forecast = await pipe.execute()

    meta    = dict(meta_row)
    current = _parse_zone_state(meta, state or _EMPTY_STATE, forecast or None)

    history = [
        ZoneHistoryPoint(
            window_start   = r["window_start"].isoformat(),
            current_demand = r["current_demand"],
            surge_signal   = float(r["surge_signal"]),
        )
        for r in history_rows
    ]

    fc = current.forecast

    return ZoneDetail(
        h3_index     = h3_index,
        centroid_lat = float(meta_row["centroid_lat"]),
        centroid_lon = float(meta_row["centroid_lon"]),
        demand_tier  = meta_row["demand_tier"],
        current      = current,
        history_24h  = history,
        forecast     = fc,
    )
