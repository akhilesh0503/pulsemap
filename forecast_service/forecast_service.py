"""
PulseMap forecast service.

Runs every FORECAST_INTERVAL_SECONDS (default 120s).
For each HIGH and MEDIUM demand zone:
  1. Reads current state from Redis zone_state:{h3_index}
  2. Reads rolling demand history from PostgreSQL zone_features
  3. Builds XGBoost feature vector
  4. Predicts next-period demand with confidence interval (±1.5 × val MAE)
  5. Writes to Redis zone_forecast:{h3_index} and PostgreSQL forecasts

Also computes PSI drift metrics on every run and writes to drift_metrics.
"""

import json
import logging
import os
import time
from datetime import datetime

import joblib
import numpy as np
import psycopg2
import psycopg2.extras
import redis as redis_lib

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

SYNC_DB_URL   = os.getenv("SYNC_DATABASE_URL", "postgresql://pulsemap:pulsemap@postgres:5432/pulsemap")
REDIS_HOST    = os.getenv("REDIS_HOST", "redis")
REDIS_PORT    = int(os.getenv("REDIS_PORT", "6379"))
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_service:5000")
MODELS_DIR    = os.getenv("MODELS_DIR", "/app/models")
INTERVAL      = int(os.getenv("FORECAST_INTERVAL_SECONDS", "120"))
PSI_THRESHOLD = float(os.getenv("PSI_ALERT_THRESHOLD", "0.2"))
FORECAST_TTL  = 300  # Redis TTL for forecast keys

FEATURE_COLS = [
    "rolling_1h_demand",
    "rolling_24h_demand",
    "rolling_7d_demand",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "avg_fare",
    "avg_distance",
]

# PSI bins for drift detection
N_BINS = 10


def _connect_redis(max_retries: int = 20, delay: int = 3) -> redis_lib.Redis:
    for attempt in range(1, max_retries + 1):
        try:
            r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            return r
        except Exception:
            log.warning("Redis not ready (%d/%d)", attempt, max_retries)
            time.sleep(delay)
    raise RuntimeError("Redis unavailable")


def _connect_postgres(max_retries: int = 20, delay: int = 3):
    for attempt in range(1, max_retries + 1):
        try:
            return psycopg2.connect(SYNC_DB_URL)
        except psycopg2.OperationalError:
            log.warning("PostgreSQL not ready (%d/%d)", attempt, max_retries)
            time.sleep(delay)
    raise RuntimeError("PostgreSQL unavailable")


def _load_models() -> tuple[dict, dict]:
    """
    Load joblib models + model_config.json from MODELS_DIR.
    Returns (models dict by tier, metadata dict).
    """
    config_path = os.path.join(MODELS_DIR, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"model_config.json not found in {MODELS_DIR} — run model-trainer first")

    with open(config_path) as f:
        config = json.load(f)

    models = {}
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        path = os.path.join(MODELS_DIR, f"model_{tier}.joblib")
        if os.path.exists(path):
            models[tier] = joblib.load(path)
            log.info("Loaded model for tier %s (MAE=%.3f)", tier,
                     config["tiers"].get(tier, {}).get("mae", -1))
        else:
            log.warning("No model file for tier %s — will use MEDIUM model as fallback", tier)

    if not models:
        raise RuntimeError("No models found — check MODELS_DIR")

    return models, config


def _load_active_zones(pg_conn) -> list[dict]:
    """Load HIGH and MEDIUM zones for forecasting."""
    cur = pg_conn.cursor()
    cur.execute("""
        SELECT h3_index, demand_tier, avg_hourly_demand, peak_hour
        FROM zone_stats
        WHERE demand_tier IN ('HIGH', 'MEDIUM')
        ORDER BY avg_hourly_demand DESC
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _get_zone_features(h3_index: str, current_hour: int,
                       current_dow: int, pg_conn, r: redis_lib.Redis) -> dict | None:
    """
    Build feature vector for one zone by combining:
    - Redis zone_state (current demand, velocity)
    - PostgreSQL zone_features (rolling historical averages)
    """
    # Current state from Redis
    state = r.hgetall(f"zone_state:{h3_index}")

    # Historical rolling averages from PostgreSQL
    cur = pg_conn.cursor()
    cur.execute("""
        SELECT rolling_1h_demand, rolling_24h_demand, rolling_7d_demand,
               avg_fare, avg_distance
        FROM zone_features
        WHERE h3_index = %s AND pickup_hour = %s
        ORDER BY window_start DESC
        LIMIT 1
    """, (h3_index, current_hour))
    hist_row = cur.fetchone()

    if not hist_row and not state:
        return None

    # Use Redis current demand as rolling_1h if available and more recent
    rolling_1h = float(state.get("current_demand", 0)) if state else 0.0
    if hist_row:
        rolling_24h = float(hist_row[1] or 0)
        rolling_7d  = float(hist_row[2] or 0)
        avg_fare     = float(hist_row[3] or 0)
        avg_distance = float(hist_row[4] or 0)
    else:
        rolling_24h = rolling_1h
        rolling_7d  = rolling_1h
        avg_fare     = 0.0
        avg_distance = 0.0

    is_weekend   = int(current_dow in [5, 6])
    is_rush_hour = int(current_hour in list(range(7, 10)) + list(range(17, 20)))

    return {
        "rolling_1h_demand":  rolling_1h,
        "rolling_24h_demand": rolling_24h,
        "rolling_7d_demand":  rolling_7d,
        "pickup_hour":         current_hour,
        "pickup_dayofweek":    current_dow,
        "is_weekend":          is_weekend,
        "is_rush_hour":        is_rush_hour,
        "avg_fare":            avg_fare,
        "avg_distance":        avg_distance,
    }


def _run_forecasts(zones: list[dict], models: dict, config: dict,
                   pg_conn, r: redis_lib.Redis) -> int:
    """Generate and persist forecasts for all active zones. Returns count."""
    now         = datetime.utcnow()
    current_hour = now.hour
    current_dow  = now.weekday()
    model_version = config.get("trained_at", "unknown")[:10]

    forecast_rows = []
    redis_pipe    = r.pipeline()
    forecasted    = 0

    for zone in zones:
        h3_idx = zone["h3_index"]
        tier   = zone["demand_tier"]

        features = _get_zone_features(h3_idx, current_hour, current_dow, pg_conn, r)
        if not features:
            continue

        model = models.get(tier) or models.get("MEDIUM")
        if not model:
            continue

        X        = np.array([[features[c] for c in FEATURE_COLS]])
        pred     = float(np.clip(model.predict(X)[0], 0, None))
        tier_meta = config.get("tiers", {}).get(tier, {})
        mae      = float(tier_meta.get("mae", pred * 0.2))
        ci_half  = mae * 1.5

        forecast_rows.append((
            h3_idx,
            now,
            15,
            round(pred, 2),
            round(max(0, pred - ci_half), 2),
            round(pred + ci_half, 2),
            model_version,
        ))

        redis_pipe.hset(f"zone_forecast:{h3_idx}", mapping={
            "predicted_demand": round(pred, 2),
            "confidence_lower": round(max(0, pred - ci_half), 2),
            "confidence_upper": round(pred + ci_half, 2),
            "forecast_time":    now.isoformat(),
            "model_version":    model_version,
        })
        redis_pipe.expire(f"zone_forecast:{h3_idx}", FORECAST_TTL)
        forecasted += 1

    # Batch write to Redis
    redis_pipe.execute()

    # Batch write to PostgreSQL
    if forecast_rows:
        cur = pg_conn.cursor()
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO forecasts
                (h3_index, forecast_time, horizon_minutes, predicted_demand,
                 confidence_lower, confidence_upper, model_version)
            VALUES %s
            """,
            forecast_rows,
            page_size=200,
        )
        pg_conn.commit()

    return forecasted


def _compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Population Stability Index.
    PSI < 0.1: no drift. 0.1–0.2: moderate drift. >0.2: significant drift.
    """
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    bins    = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    b_counts = np.histogram(baseline, bins=bins)[0] / len(baseline)
    c_counts = np.histogram(current,  bins=bins)[0] / len(current)

    # Avoid zeros
    b_counts = np.clip(b_counts, 1e-6, None)
    c_counts = np.clip(c_counts, 1e-6, None)

    return float(np.sum((c_counts - b_counts) * np.log(c_counts / b_counts)))


def _run_drift_detection(pg_conn) -> int:
    """
    Compute PSI for rolling_1h_demand comparing:
    - baseline: same hour, 7 days ago  ±3 days
    - current:  same hour, last 2 hours

    Writes results to drift_metrics. Returns number of features computed.
    """
    now  = datetime.utcnow()
    hour = now.hour

    cur = pg_conn.cursor()

    # Baseline: same hour, 7-day window ending 6 days ago
    cur.execute("""
        SELECT trip_count, rolling_1h_demand, avg_fare
        FROM zone_features
        WHERE pickup_hour = %s
          AND window_start BETWEEN NOW() - INTERVAL '10 days'
                               AND NOW() - INTERVAL '3 days'
    """, (hour,))
    baseline_rows = cur.fetchall()

    # Current: same hour, last 2 hours
    cur.execute("""
        SELECT current_demand
        FROM streaming_events
        WHERE EXTRACT(HOUR FROM window_start) = %s
          AND window_start >= NOW() - INTERVAL '2 hours'
    """, (hour,))
    current_rows = cur.fetchall()

    if not baseline_rows or not current_rows:
        return 0

    baseline_demand = np.array([float(r[0]) for r in baseline_rows])
    current_demand  = np.array([float(r[0]) for r in current_rows])
    baseline_rolling = np.array([float(r[1] or 0) for r in baseline_rows])
    baseline_fare    = np.array([float(r[2] or 0) for r in baseline_rows if r[2]])

    features_to_check = [
        ("trip_count",        baseline_demand,  current_demand),
        ("rolling_1h_demand", baseline_rolling, current_demand),
    ]

    rows = []
    for fname, bdata, cdata in features_to_check:
        psi  = _compute_psi(bdata, cdata)
        rows.append((
            now, fname,
            round(psi, 6),
            round(float(np.mean(bdata)), 4),
            round(float(np.mean(cdata)), 4),
            psi > PSI_THRESHOLD,
        ))
        if psi > PSI_THRESHOLD:
            log.warning("Drift alert: %s PSI=%.4f exceeds threshold %.2f", fname, psi, PSI_THRESHOLD)

    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO drift_metrics
            (computed_at, feature_name, psi_score, baseline_mean, current_mean, alert_triggered)
        VALUES %s
        """,
        rows,
    )
    pg_conn.commit()
    return len(rows)


def main():
    log.info("=== PulseMap forecast service starting (interval=%ds) ===", INTERVAL)

    # Wait for models to be available (model-trainer runs first)
    models, config = None, None
    for attempt in range(60):
        try:
            models, config = _load_models()
            break
        except FileNotFoundError:
            log.warning("Models not ready yet (attempt %d/60) — waiting 10s", attempt + 1)
            time.sleep(10)

    if models is None:
        raise RuntimeError("Models never became available — ensure model-trainer completed")

    r       = _connect_redis()
    pg_conn = _connect_postgres()
    zones   = _load_active_zones(pg_conn)
    log.info("Loaded %d active zones for forecasting", len(zones))

    run = 0
    while True:
        run += 1
        start = time.monotonic()

        try:
            # Refresh zone list hourly
            if run % 30 == 0:
                zones = _load_active_zones(pg_conn)

            forecasted = _run_forecasts(zones, models, config, pg_conn, r)
            drift_count = _run_drift_detection(pg_conn)

            elapsed = time.monotonic() - start
            log.info("Run %d: %d forecasts, %d drift features — %.2fs",
                     run, forecasted, drift_count, elapsed)
        except psycopg2.OperationalError:
            log.warning("PostgreSQL error — reconnecting")
            pg_conn = _connect_postgres(max_retries=5)
        except Exception as exc:
            log.error("Forecast run %d failed: %s", run, exc)

        time.sleep(max(0, INTERVAL - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
