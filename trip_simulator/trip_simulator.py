"""
PulseMap trip simulator.

Replays NYC TLC yellow taxi trips from parquet files at configurable
speed (default 60x: 1 real hour = 60 seconds) and publishes each trip
as a JSON event to the Kafka raw_trips topic.

Falls back to synthetic event generation using zone_stats from
PostgreSQL if parquet files are absent.

Loops continuously — when all trips are exhausted, replay restarts
from the beginning.
"""

import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

KAFKA_SERVERS   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_RAW       = os.getenv("KAFKA_TOPIC_RAW_TRIPS", "raw_trips")
DATA_DIR        = os.getenv("DATA_DIR", "/app/data")
SYNC_DB_URL     = os.getenv("SYNC_DATABASE_URL", "postgresql://pulsemap:pulsemap@postgres:5432/pulsemap")
SPEED_MULT      = float(os.getenv("SIMULATOR_SPEED_MULTIPLIER", "60"))
H3_RESOLUTION   = int(os.getenv("H3_RESOLUTION", "8"))

PARQUET_FILES = [
    os.path.join(DATA_DIR, "yellow_tripdata_2024-01.parquet"),
    os.path.join(DATA_DIR, "yellow_tripdata_2024-02.parquet"),
]

NYC_LON_MIN, NYC_LON_MAX = -74.05, -73.75
NYC_LAT_MIN, NYC_LAT_MAX =  40.63,  40.85


def _wait_for_kafka(max_retries: int = 30, delay: int = 5) -> KafkaProducer:
    for attempt in range(1, max_retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
                max_block_ms=10_000,
            )
            log.info("Connected to Kafka at %s", KAFKA_SERVERS)
            return producer
        except NoBrokersAvailable:
            log.warning("Kafka not ready (attempt %d/%d) — retrying in %ds", attempt, max_retries, delay)
            time.sleep(delay)
    raise RuntimeError("Kafka unavailable after retries")


def _load_trips() -> pd.DataFrame | None:
    import h3
    found = [p for p in PARQUET_FILES if os.path.exists(p)]
    if not found:
        return None

    dfs = []
    for path in found:
        log.info("Loading %s for simulation", path)
        df = pd.read_parquet(path)

        # Normalise datetime column name across TLC schema versions
        col_map = {}
        for col in df.columns:
            lc = col.lower()
            if "pickup_datetime" in lc or "tpep_pickup" in lc:
                col_map[col] = "pickup_datetime"
            elif lc in ("pickup_longitude", "start_lon"):
                col_map[col] = "pickup_longitude"
            elif lc in ("pickup_latitude", "start_lat"):
                col_map[col] = "pickup_latitude"
            elif lc == "fare_amount":
                col_map[col] = "fare_amount"
            elif lc == "passenger_count":
                col_map[col] = "passenger_count"
            elif lc in ("dropoff_longitude", "end_lon"):
                col_map[col] = "dropoff_longitude"
            elif lc in ("dropoff_latitude", "end_lat"):
                col_map[col] = "dropoff_latitude"
        df = df.rename(columns=col_map)

        # Post-2016 TLC files use LocationID — synthesise coordinates
        if "pickup_longitude" not in df.columns:
            log.warning("%s uses LocationID schema — synthesising coordinates", path)
            rng = np.random.default_rng(42)
            n = len(df)
            df["pickup_longitude"]  = rng.uniform(NYC_LON_MIN, NYC_LON_MAX, n)
            df["pickup_latitude"]   = rng.uniform(NYC_LAT_MIN, NYC_LAT_MAX, n)
            df["dropoff_longitude"] = rng.uniform(NYC_LON_MIN, NYC_LON_MAX, n)
            df["dropoff_latitude"]  = rng.uniform(NYC_LAT_MIN, NYC_LAT_MAX, n)

        if "pickup_datetime" not in df.columns:
            log.error("Cannot find pickup datetime column in %s — skipping", path)
            continue

        keep_cols = ["pickup_datetime", "pickup_latitude", "pickup_longitude"]
        optional  = ["dropoff_latitude", "dropoff_longitude", "fare_amount", "passenger_count"]
        keep_cols += [c for c in optional if c in df.columns]
        dfs.append(df[keep_cols].copy())

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Filter to NYC bbox
    mask = (
        combined["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX) &
        combined["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)
    )
    combined = combined[mask].dropna(subset=["pickup_datetime"]).copy()
    combined["pickup_datetime"] = pd.to_datetime(combined["pickup_datetime"]).dt.tz_localize(None)
    combined = combined.sort_values("pickup_datetime").reset_index(drop=True)

    # Add H3 indexes
    log.info("Computing H3 indexes for %d trips...", len(combined))
    combined["pickup_h3"] = combined.apply(
        lambda r: h3.geo_to_h3(r["pickup_latitude"], r["pickup_longitude"], H3_RESOLUTION),
        axis=1,
    )
    if "dropoff_latitude" in combined.columns:
        combined["dropoff_h3"] = combined.apply(
            lambda r: h3.geo_to_h3(r["dropoff_latitude"], r["dropoff_longitude"], H3_RESOLUTION),
            axis=1,
        )
    else:
        combined["dropoff_h3"] = combined["pickup_h3"]

    log.info("Loaded %d trips for simulation (%d unique pickup zones)",
             len(combined), combined["pickup_h3"].nunique())
    return combined


def _load_zones_from_db() -> list[str]:
    """Load active H3 zone indexes from zone_stats for synthetic fallback."""
    try:
        conn = psycopg2.connect(SYNC_DB_URL)
        cur  = conn.cursor()
        cur.execute("SELECT h3_index FROM zone_stats")
        zones = [row[0] for row in cur.fetchall()]
        conn.close()
        log.info("Loaded %d zones from zone_stats for synthetic simulation", len(zones))
        return zones
    except Exception as exc:
        log.warning("Could not load zones from DB (%s) — using fallback zone list", exc)
        return []


def _make_synthetic_event(rng: np.random.Generator, zones: list[str], sim_ts: datetime) -> dict:
    import h3
    if zones:
        pickup_h3  = rng.choice(zones)
        lat, lon   = h3.h3_to_geo(pickup_h3)
        dropoff_h3 = rng.choice(zones)
    else:
        lat  = rng.uniform(NYC_LAT_MIN, NYC_LAT_MAX)
        lon  = rng.uniform(NYC_LON_MIN, NYC_LON_MAX)
        import h3 as _h3
        pickup_h3  = _h3.geo_to_h3(lat, lon, H3_RESOLUTION)
        dropoff_h3 = pickup_h3

    return {
        "pickup_h3":       pickup_h3,
        "dropoff_h3":      dropoff_h3,
        "pickup_lat":      round(float(lat), 6),
        "pickup_lon":      round(float(lon), 6),
        "pickup_ts":       sim_ts.isoformat(),
        "fare_amount":     round(float(rng.lognormal(2.8, 0.6)), 2),
        "passenger_count": int(rng.choice([1, 1, 1, 2, 2, 3])),
    }


def _simulate_from_parquet(trips: pd.DataFrame, producer: KafkaProducer) -> None:
    """Replay parquet trips at SPEED_MULT × real speed."""
    log.info("Starting parquet replay at %gx speed (%d trips)", SPEED_MULT, len(trips))

    sim_start  = trips["pickup_datetime"].iloc[0]
    wall_start = time.monotonic()
    published  = 0

    for idx, row in trips.iterrows():
        # How far into the simulation is this event (simulated seconds)?
        sim_offset_s = (row["pickup_datetime"] - sim_start).total_seconds()
        # How many real seconds should have elapsed?
        real_offset_s = sim_offset_s / SPEED_MULT

        elapsed = time.monotonic() - wall_start
        wait    = real_offset_s - elapsed
        if wait > 0:
            time.sleep(wait)

        event = {
            "pickup_h3":       row["pickup_h3"],
            "dropoff_h3":      row["dropoff_h3"],
            "pickup_lat":      round(float(row["pickup_latitude"]), 6),
            "pickup_lon":      round(float(row["pickup_longitude"]), 6),
            "pickup_ts":       row["pickup_datetime"].isoformat(),
            "fare_amount":     round(float(row.get("fare_amount", 0) or 0), 2),
            "passenger_count": int(row.get("passenger_count", 1) or 1),
        }
        producer.send(TOPIC_RAW, value=event)
        published += 1

        if published % 10_000 == 0:
            simulated_time = row["pickup_datetime"].strftime("%Y-%m-%d %H:%M")
            log.info("Published %d trips — simulated time: %s", published, simulated_time)

    producer.flush()
    log.info("Replay complete — %d trips published. Looping.", published)


def _simulate_synthetic(producer: KafkaProducer) -> None:
    """Synthetic fallback: generate events based on hourly demand curve."""
    log.info("Starting synthetic simulation (no parquet data)")
    rng   = np.random.default_rng(seed=int(time.time()))
    zones = _load_zones_from_db()

    # Hourly demand weights (same as batch pipeline synthetic generator)
    hour_weights = np.array([
        0.3, 0.2, 0.15, 0.1, 0.15, 0.4,
        0.8, 1.4, 1.6, 1.2, 0.9, 1.0,
        1.1, 1.0, 0.9, 1.0, 1.3, 1.7,
        1.8, 1.5, 1.2, 0.9, 0.7, 0.5,
    ])
    # Max trips per second at peak (rush hour ~6pm NYC)
    base_rate = 2.0  # trips/second at weight=1.0

    published = 0
    while True:
        now  = datetime.utcnow()
        hour = now.hour
        rate = base_rate * hour_weights[hour]

        # Poisson-distributed inter-arrival times
        interval = rng.exponential(1.0 / max(rate, 0.1))
        time.sleep(interval)

        event = _make_synthetic_event(rng, zones, now)
        producer.send(TOPIC_RAW, value=event)
        published += 1

        if published % 1000 == 0:
            log.info("Synthetic: published %d events", published)


def main():
    log.info("=== PulseMap trip simulator starting (speed=%gx) ===", SPEED_MULT)
    producer = _wait_for_kafka()

    trips = _load_trips()

    if trips is not None:
        while True:
            _simulate_from_parquet(trips, producer)
            log.info("Loop restart — replaying from beginning")
    else:
        log.warning("No parquet files found — switching to synthetic event generation")
        _simulate_synthetic(producer)


if __name__ == "__main__":
    main()
