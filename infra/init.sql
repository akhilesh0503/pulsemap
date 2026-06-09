-- PulseMap database schema

-- Zone stats: one row per active H3 zone, computed by batch pipeline
CREATE TABLE IF NOT EXISTS zone_stats (
    h3_index            VARCHAR(20)      PRIMARY KEY,
    centroid_lat        DOUBLE PRECISION NOT NULL,
    centroid_lon        DOUBLE PRECISION NOT NULL,
    total_trips         BIGINT           NOT NULL DEFAULT 0,
    avg_hourly_demand   DOUBLE PRECISION,
    peak_hour           INT,
    demand_tier         VARCHAR(10)      NOT NULL DEFAULT 'LOW',
    updated_at          TIMESTAMPTZ      DEFAULT NOW()
);

-- Zone features: hourly batch-computed feature vectors per zone
CREATE TABLE IF NOT EXISTS zone_features (
    id                   BIGSERIAL        PRIMARY KEY,
    h3_index             VARCHAR(20)      NOT NULL,
    window_start         TIMESTAMPTZ      NOT NULL,
    trip_count           INT              NOT NULL DEFAULT 0,
    avg_fare             DOUBLE PRECISION,
    avg_distance         DOUBLE PRECISION,
    avg_passenger_count  DOUBLE PRECISION,
    pickup_hour          INT,
    pickup_dayofweek     INT,
    is_weekend           BOOLEAN,
    is_rush_hour         BOOLEAN,
    rolling_1h_demand    DOUBLE PRECISION,
    rolling_24h_demand   DOUBLE PRECISION,
    rolling_7d_demand    DOUBLE PRECISION,
    created_at           TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_zone_features_h3     ON zone_features(h3_index);
CREATE INDEX IF NOT EXISTS idx_zone_features_window ON zone_features(window_start);
CREATE UNIQUE INDEX IF NOT EXISTS idx_zone_features_h3_window ON zone_features(h3_index, window_start);

-- Streaming events: 2-minute tumbling window output from stream processor
CREATE TABLE IF NOT EXISTS streaming_events (
    id                    BIGSERIAL        PRIMARY KEY,
    h3_index              VARCHAR(20)      NOT NULL,
    window_start          TIMESTAMPTZ      NOT NULL,
    window_end            TIMESTAMPTZ      NOT NULL,
    current_demand        INT              NOT NULL DEFAULT 0,
    demand_velocity       DOUBLE PRECISION,
    supply_pressure_index DOUBLE PRECISION,
    surge_signal          DOUBLE PRECISION,
    created_at            TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_streaming_events_h3     ON streaming_events(h3_index);
CREATE INDEX IF NOT EXISTS idx_streaming_events_window ON streaming_events(window_start);

-- Forecasts: XGBoost 15-minute ahead demand predictions
CREATE TABLE IF NOT EXISTS forecasts (
    id                BIGSERIAL        PRIMARY KEY,
    h3_index          VARCHAR(20)      NOT NULL,
    forecast_time     TIMESTAMPTZ      NOT NULL,
    horizon_minutes   INT              NOT NULL DEFAULT 15,
    predicted_demand  DOUBLE PRECISION,
    confidence_lower  DOUBLE PRECISION,
    confidence_upper  DOUBLE PRECISION,
    model_version     VARCHAR(100),
    created_at        TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_forecasts_h3   ON forecasts(h3_index);
CREATE INDEX IF NOT EXISTS idx_forecasts_time ON forecasts(forecast_time);

-- Drift metrics: PSI scores vs 7-day baseline, updated by forecast service
CREATE TABLE IF NOT EXISTS drift_metrics (
    id              BIGSERIAL        PRIMARY KEY,
    computed_at     TIMESTAMPTZ      NOT NULL,
    feature_name    VARCHAR(100)     NOT NULL,
    psi_score       DOUBLE PRECISION,
    baseline_mean   DOUBLE PRECISION,
    current_mean    DOUBLE PRECISION,
    alert_triggered BOOLEAN          NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_time    ON drift_metrics(computed_at);
CREATE INDEX IF NOT EXISTS idx_drift_metrics_feature ON drift_metrics(feature_name);
