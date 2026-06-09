"""
PulseMap stream processor.

Implements Flink windowing semantics in Python:
  - Tumbling 2-minute windows per H3 zone (processing-time based)
  - Computes demand, surge signal, and supply pressure index per window
  - Writes results to Redis (zone_state), PostgreSQL (streaming_events),
    and Kafka (zone_updates topic for SSE)

Production deployment note: this would use PyFlink for horizontal
scale, exactly-once guarantees via Kafka checkpointing, and event-time
watermarks. The Python implementation captures the same logic and
produces identical outputs for a single-node deployment.
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import redis as redis_lib
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

KAFKA_SERVERS     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_RAW         = os.getenv("KAFKA_TOPIC_RAW_TRIPS", "raw_trips")
TOPIC_UPDATES     = os.getenv("KAFKA_TOPIC_ZONE_UPDATES", "zone_updates")
CONSUMER_GROUP    = os.getenv("KAFKA_CONSUMER_GROUP_STREAM", "stream-processor-group")
SYNC_DB_URL       = os.getenv("SYNC_DATABASE_URL", "postgresql://pulsemap:pulsemap@postgres:5432/pulsemap")
REDIS_HOST        = os.getenv("REDIS_HOST", "redis")
REDIS_PORT        = int(os.getenv("REDIS_PORT", "6379"))
WINDOW_SECONDS    = int(os.getenv("STREAM_WINDOW_MINUTES", "2")) * 60
SURGE_CAP         = float(os.getenv("SURGE_CAP", "3.5"))
ZONE_STATE_TTL    = 300   # Redis TTL for zone_state keys: 5 minutes

STATS_KEY         = "processor:stats"
REDIS_HIT_KEY     = "processor:redis_ops"


def _wait_for_kafka(max_retries: int = 30, delay: int = 5):
    for attempt in range(1, max_retries + 1):
        try:
            consumer = KafkaConsumer(
                TOPIC_RAW,
                bootstrap_servers=KAFKA_SERVERS,
                group_id=CONSUMER_GROUP,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms=1000,
                max_poll_records=500,
            )
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks=1,
            )
            log.info("Connected to Kafka")
            return consumer, producer
        except NoBrokersAvailable:
            log.warning("Kafka not ready (attempt %d/%d) — retrying in %ds", attempt, max_retries, delay)
            time.sleep(delay)
    raise RuntimeError("Kafka unavailable after retries")


def _connect_redis(max_retries: int = 20, delay: int = 3) -> redis_lib.Redis:
    for attempt in range(1, max_retries + 1):
        try:
            r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            log.info("Connected to Redis")
            return r
        except Exception:
            log.warning("Redis not ready (attempt %d/%d)", attempt, max_retries)
            time.sleep(delay)
    raise RuntimeError("Redis unavailable after retries")


def _connect_postgres(max_retries: int = 20, delay: int = 3):
    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(SYNC_DB_URL)
            log.info("Connected to PostgreSQL")
            return conn
        except psycopg2.OperationalError:
            log.warning("PostgreSQL not ready (attempt %d/%d)", attempt, max_retries)
            time.sleep(delay)
    raise RuntimeError("PostgreSQL unavailable after retries")


def _load_historical_averages(pg_conn) -> dict:
    """
    Load per-zone hourly average demand from zone_features.
    Returns {(h3_index, hour): avg_trip_count}.
    Used to compute supply_pressure_index.
    """
    cur = pg_conn.cursor()
    cur.execute("""
        SELECT h3_index, pickup_hour, AVG(trip_count) AS avg_demand
        FROM zone_features
        GROUP BY h3_index, pickup_hour
    """)
    rows = cur.fetchall()
    hist = {(row[0], row[1]): float(row[2]) for row in rows}
    log.info("Loaded historical averages for %d (zone, hour) pairs", len(hist))
    return hist


def _compute_surge(current_demand: int, h3_index: str, hour: int,
                   hist_avg: dict) -> tuple[float, float]:
    """
    supply_pressure_index = current_demand / historical_avg_for_zone_this_hour
    surge_signal = 1.0 + max(0, spi - 1.0) * 0.8, capped at SURGE_CAP

    Returns (supply_pressure_index, surge_signal).
    """
    hist = hist_avg.get((h3_index, hour), None)
    if hist is None or hist < 0.1:
        # No history for this zone/hour — use window-relative baseline
        hist = max(current_demand, 1.0)

    spi   = current_demand / hist
    surge = 1.0 + max(0.0, spi - 1.0) * 0.8
    surge = min(surge, SURGE_CAP)
    return round(spi, 4), round(surge, 4)


def _write_redis(r: redis_lib.Redis, results: list[dict]) -> None:
    pipe = r.pipeline()
    for res in results:
        key = f"zone_state:{res['h3_index']}"
        pipe.hset(key, mapping={
            "current_demand":        res["current_demand"],
            "demand_velocity":       res["demand_velocity"],
            "supply_pressure_index": res["supply_pressure_index"],
            "surge_signal":          res["surge_signal"],
            "window_start":          res["window_start"].isoformat(),
            "window_end":            res["window_end"].isoformat(),
        })
        pipe.expire(key, ZONE_STATE_TTL)
    pipe.execute()
    # Track Redis write count for pipeline health endpoint
    r.incrby(REDIS_HIT_KEY, len(results))


def _write_postgres(pg_conn, results: list[dict]) -> None:
    cur = pg_conn.cursor()
    rows = [
        (
            r["h3_index"],
            r["window_start"],
            r["window_end"],
            r["current_demand"],
            r["demand_velocity"],
            r["supply_pressure_index"],
            r["surge_signal"],
        )
        for r in results
    ]
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO streaming_events
            (h3_index, window_start, window_end, current_demand,
             demand_velocity, supply_pressure_index, surge_signal)
        VALUES %s
        """,
        rows,
        page_size=200,
    )
    pg_conn.commit()


def _write_kafka(producer: KafkaProducer, results: list[dict]) -> None:
    for res in results:
        producer.send(TOPIC_UPDATES, value=res)
    producer.flush()


def _update_processor_stats(r: redis_lib.Redis, windows_processed: int,
                             total_events: int, zones_count: int) -> None:
    r.hset(STATS_KEY, mapping={
        "last_window_time":    datetime.utcnow().isoformat(),
        "windows_processed":   windows_processed,
        "total_events":        total_events,
        "last_zones_count":    zones_count,
        "window_seconds":      WINDOW_SECONDS,
    })


def main():
    log.info("=== PulseMap stream processor starting (window=%ds) ===", WINDOW_SECONDS)

    consumer, producer = _wait_for_kafka()
    r        = _connect_redis()
    pg_conn  = _connect_postgres()
    hist_avg = _load_historical_averages(pg_conn)

    # Refresh historical averages every hour
    hist_loaded_at = time.monotonic()
    HIST_REFRESH_INTERVAL = 3600

    # Window state: {h3_index: trip_count_in_current_window}
    window_buffer:   dict[str, int]   = defaultdict(int)
    previous_demand: dict[str, int]   = {}
    last_flush       = time.monotonic()
    windows_done     = 0
    total_events     = 0

    log.info("Stream processor ready — consuming from '%s'", TOPIC_RAW)

    while True:
        # ── Consume messages ──────────────────────────────────────────────
        try:
            msg_batch = consumer.poll(timeout_ms=500, max_records=500)
        except Exception as exc:
            log.error("Kafka poll error: %s", exc)
            time.sleep(2)
            continue

        for _tp, records in msg_batch.items():
            for record in records:
                event = record.value
                h3_idx = event.get("pickup_h3")
                if h3_idx:
                    window_buffer[h3_idx] += 1
                    total_events += 1

        # ── Flush window when time is up ──────────────────────────────────
        now = time.monotonic()
        if now - last_flush >= WINDOW_SECONDS:
            window_start = datetime.utcnow().replace(
                second=0, microsecond=0,
            ).replace(tzinfo=None) - __import__("datetime").timedelta(seconds=WINDOW_SECONDS)
            window_end = datetime.utcnow().replace(second=0, microsecond=0).replace(tzinfo=None)
            current_hour = window_end.hour

            results = []
            for h3_idx, count in window_buffer.items():
                prev     = previous_demand.get(h3_idx, 0)
                spi, surge = _compute_surge(count, h3_idx, current_hour, hist_avg)
                results.append({
                    "h3_index":              h3_idx,
                    "window_start":          window_start,
                    "window_end":            window_end,
                    "current_demand":        count,
                    "demand_velocity":       count - prev,
                    "supply_pressure_index": spi,
                    "surge_signal":          surge,
                })
                previous_demand[h3_idx] = count

            if results:
                try:
                    _write_redis(r, results)
                    _write_postgres(pg_conn, results)
                    _write_kafka(producer, results)
                except psycopg2.OperationalError:
                    log.warning("PostgreSQL write failed — reconnecting")
                    pg_conn = _connect_postgres(max_retries=5)

            windows_done += 1
            _update_processor_stats(r, windows_done, total_events, len(results))

            log.info("Window %d: %d active zones, %d events — surge max=%.2f",
                     windows_done, len(results), total_events,
                     max((r["surge_signal"] for r in results), default=1.0))

            # Reset for next window
            window_buffer = defaultdict(int)
            last_flush    = now

        # ── Refresh historical averages hourly ────────────────────────────
        if now - hist_loaded_at >= HIST_REFRESH_INTERVAL:
            hist_avg       = _load_historical_averages(pg_conn)
            hist_loaded_at = now


if __name__ == "__main__":
    main()
