"""
PulseMap demand forecasting model trainer.

Trains one XGBoost regressor per demand tier (HIGH / MEDIUM / LOW).
Target: next-hour trip count for a given H3 zone.

Train split: January 2024  (window_start < 2024-02-01)
Val split:   February 2024 week 1 (2024-02-01 to 2024-02-07)

MLflow logs params + metrics + feature importances per tier.
Models saved to MODELS_DIR as joblib files for direct use by
forecast_service (avoids MLflow dependency at inference time).
Also registered in MLflow model registry for lineage tracking.
"""

import json
import logging
import os
import sys
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger(__name__)

SYNC_DB_URL  = os.getenv("SYNC_DATABASE_URL", "postgresql://pulsemap:pulsemap@postgres:5432/pulsemap")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_service:5000")
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT_NAME", "pulsemap-demand-forecast")
MODELS_DIR   = os.getenv("MODELS_DIR", "/app/models")

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

TRAIN_CUTOFF = datetime(2024, 2, 1)
VAL_END      = datetime(2024, 2, 8)

XGB_PARAMS = {
    "n_estimators":  400,
    "max_depth":     5,
    "learning_rate": 0.05,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "random_state":  42,
    "n_jobs":        -1,
    "tree_method":   "hist",
}


def _load_features() -> pd.DataFrame:
    conn = psycopg2.connect(SYNC_DB_URL)
    cur  = conn.cursor()
    cur.execute("""
        SELECT zf.h3_index, zf.window_start, zf.trip_count,
               zf.avg_fare, zf.avg_distance, zf.avg_passenger_count,
               zf.pickup_hour, zf.pickup_dayofweek, zf.is_weekend,
               zf.is_rush_hour, zf.rolling_1h_demand,
               zf.rolling_24h_demand, zf.rolling_7d_demand,
               zs.demand_tier
        FROM zone_features zf
        JOIN zone_stats zs ON zs.h3_index = zf.h3_index
        ORDER BY zf.h3_index, zf.window_start
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        log.error("zone_features is empty — batch pipeline must run before model trainer")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=[
        "h3_index", "window_start", "trip_count",
        "avg_fare", "avg_distance", "avg_passenger_count",
        "pickup_hour", "pickup_dayofweek", "is_weekend",
        "is_rush_hour", "rolling_1h_demand",
        "rolling_24h_demand", "rolling_7d_demand",
        "demand_tier",
    ])
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True).dt.tz_localize(None)
    log.info("Loaded %d zone-hour rows across %d zones", len(df), df["h3_index"].nunique())
    return df


def _build_training_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create supervised learning target: next-hour trip count per zone.
    Shifts trip_count by -1 within each zone's time series.
    """
    df = df.sort_values(["h3_index", "window_start"]).copy()

    # Create target: next window's trip count for this zone
    df["target"] = df.groupby("h3_index")["trip_count"].shift(-1)

    # Drop last row per zone (no next window) and any NaN targets
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(float)

    # Fill missing optional features with 0
    for col in ["avg_fare", "avg_distance"]:
        df[col] = df[col].fillna(0.0)
    df["is_weekend"]   = df["is_weekend"].astype(int)
    df["is_rush_hour"] = df["is_rush_hour"].astype(int)

    log.info("Training set: %d samples after target creation", len(df))
    return df


def _split(df: pd.DataFrame):
    train = df[df["window_start"] < TRAIN_CUTOFF]
    val   = df[(df["window_start"] >= TRAIN_CUTOFF) & (df["window_start"] < VAL_END)]
    return train, val


def _train_tier(tier: str, train_df: pd.DataFrame, val_df: pd.DataFrame, experiment_id: str) -> dict:
    """Train and evaluate one XGBoost model for a demand tier. Returns metrics."""
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["target"].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df["target"].values

    log.info("Tier %s — train: %d samples, val: %d samples", tier, len(X_train), len(X_val))

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"xgb-{tier.lower()}"):
        mlflow.log_params({**XGB_PARAMS, "tier": tier, "n_features": len(FEATURE_COLS)})

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds     = model.predict(X_val)
        preds     = np.clip(preds, 0, None)
        mae       = float(mean_absolute_error(y_val, preds))
        rmse      = float(np.sqrt(mean_squared_error(y_val, preds)))
        r2        = float(r2_score(y_val, preds))

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

        # Feature importances
        importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
        mlflow.log_params({f"imp_{k}": round(v, 4) for k, v in importances.items()})

        log.info("Tier %s — MAE=%.3f RMSE=%.3f R²=%.3f", tier, mae, rmse, r2)

        # Save model as MLflow artifact + register
        model_name = f"pulsemap-demand-{tier}"
        mlflow.sklearn.log_model(model, artifact_path="model",
                                  registered_model_name=model_name)

        run_id = mlflow.active_run().info.run_id

    return {
        "tier":       tier,
        "mae":        mae,
        "rmse":       rmse,
        "r2":         r2,
        "run_id":     run_id,
        "model":      model,
        "importances": importances,
    }


def _save_joblib(results: list[dict]) -> None:
    """Save joblib files + metadata to MODELS_DIR for forecast_service."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    metadata = {
        "trained_at":    datetime.utcnow().isoformat(),
        "feature_cols":  FEATURE_COLS,
        "train_cutoff":  TRAIN_CUTOFF.isoformat(),
        "val_end":       VAL_END.isoformat(),
        "tiers":         {},
    }

    for r in results:
        tier      = r["tier"]
        path      = os.path.join(MODELS_DIR, f"model_{tier}.joblib")
        joblib.dump(r["model"], path)
        log.info("Saved %s → %s", tier, path)
        metadata["tiers"][tier] = {
            "mae":        r["mae"],
            "rmse":       r["rmse"],
            "r2":         r["r2"],
            "run_id":     r["run_id"],
            "importances": r["importances"],
        }

    meta_path = os.path.join(MODELS_DIR, "model_config.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved model metadata → %s", meta_path)


def main():
    log.info("=== PulseMap model trainer starting ===")

    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment = mlflow.set_experiment(EXPERIMENT)
    experiment_id = experiment.experiment_id

    df = _load_features()
    df = _build_training_set(df)

    train_df, val_df = _split(df)
    log.info("Train rows: %d | Val rows: %d", len(train_df), len(val_df))

    if len(val_df) == 0:
        log.warning("No validation data (need Feb 2024 data) — using 10%% holdout from train")
        split_idx = int(len(train_df) * 0.9)
        val_df    = train_df.iloc[split_idx:]
        train_df  = train_df.iloc[:split_idx]

    tiers   = ["HIGH", "MEDIUM", "LOW"]
    results = []

    for tier in tiers:
        tier_train = train_df[train_df["demand_tier"] == tier]
        tier_val   = val_df[val_df["demand_tier"] == tier]

        if len(tier_train) < 100:
            log.warning("Tier %s has only %d training rows — using all tiers combined", tier, len(tier_train))
            tier_train = train_df
            tier_val   = val_df

        result = _train_tier(tier, tier_train, tier_val, experiment_id)
        results.append(result)

    _save_joblib(results)

    log.info("=== Model training complete ===")
    for r in results:
        log.info("  %s — MAE=%.3f RMSE=%.3f R²=%.3f run_id=%s",
                 r["tier"], r["mae"], r["rmse"], r["r2"], r["run_id"])


if __name__ == "__main__":
    main()
