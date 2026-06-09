from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL:                str   = "postgresql://pulsemap:pulsemap@postgres:5432/pulsemap"
    SYNC_DATABASE_URL:           str   = "postgresql://pulsemap:pulsemap@postgres:5432/pulsemap"
    REDIS_URL:                   str   = "redis://redis:6379"
    REDIS_HOST:                  str   = "redis"
    REDIS_PORT:                  int   = 6379
    KAFKA_BOOTSTRAP_SERVERS:     str   = "kafka:9092"
    KAFKA_TOPIC_ZONE_UPDATES:    str   = "zone_updates"
    KAFKA_TOPIC_RAW_TRIPS:       str   = "raw_trips"
    KAFKA_CONSUMER_GROUP_STREAM: str   = "stream-processor-group"
    H3_RESOLUTION:               int   = 8
    SURGE_CAP:                   float = 3.5
    LOG_LEVEL:                   str   = "INFO"

    class Config:
        env_file = ".env"
        extra    = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
