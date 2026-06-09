import redis.asyncio as aioredis

_redis: aioredis.Redis | None = None


def init_redis(url: str) -> None:
    global _redis
    _redis = aioredis.from_url(url, decode_responses=True)


async def close_redis() -> None:
    if _redis:
        await _redis.aclose()


def get_redis() -> aioredis.Redis:
    assert _redis is not None, "Redis not initialised"
    return _redis
