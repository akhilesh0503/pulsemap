from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ZoneForecast(BaseModel):
    predicted_demand:  float
    confidence_lower:  float
    confidence_upper:  float
    forecast_time:     Optional[str] = None
    model_version:     Optional[str] = None


class ZoneState(BaseModel):
    h3_index:              str
    centroid_lat:          float
    centroid_lon:          float
    demand_tier:           str
    current_demand:        int
    demand_velocity:       int
    supply_pressure_index: float
    surge_signal:          float
    window_start:          Optional[str] = None
    window_end:            Optional[str] = None
    forecast:              Optional[ZoneForecast] = None


class ZoneHistoryPoint(BaseModel):
    window_start:  str
    current_demand: int
    surge_signal:   float


class ZoneDetail(BaseModel):
    h3_index:     str
    centroid_lat: float
    centroid_lon: float
    demand_tier:  str
    current:      ZoneState
    history_24h:  list[ZoneHistoryPoint]
    forecast:     Optional[ZoneForecast] = None


class PipelineMetrics(BaseModel):
    kafka_consumer_lag:          int
    seconds_since_last_window:   float
    redis_writes_total:          int
    seconds_since_last_forecast: float
    total_events_processed:      int
    active_zones_last_window:    int
    stream_processor_status:     str


class DemandMetrics(BaseModel):
    total_active_zones:  int
    zones_in_surge:      int
    avg_surge_signal:    float
    max_surge_signal:    float
    highest_demand_zone: Optional[str]
    demand_vs_baseline:  float


class DriftFeature(BaseModel):
    feature_name:    str
    psi_score:       float
    baseline_mean:   float
    current_mean:    float
    alert_triggered: bool


class DriftMetrics(BaseModel):
    computed_at: str
    features:    list[DriftFeature]


class HealthResponse(BaseModel):
    status:    str
    postgres:  str
    redis:     str
    kafka:     str
    timestamp: str
