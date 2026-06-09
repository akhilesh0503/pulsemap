"""
GET /api/events/stream  — SSE endpoint consuming Kafka zone_updates topic.
Each zone update from the stream processor is forwarded to connected
browser clients as a Server-Sent Event.
"""

import json
import logging

from aiokafka import AIOKafkaConsumer
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.config import get_settings

log      = logging.getLogger(__name__)
router   = APIRouter(prefix="/api/events", tags=["events"])
settings = get_settings()


@router.get("/stream")
async def stream_zone_updates():
    async def event_generator():
        consumer = AIOKafkaConsumer(
            settings.KAFKA_TOPIC_ZONE_UPDATES,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset="latest",
            group_id=None,  # independent consumer per SSE connection
        )
        await consumer.start()
        log.info("SSE client connected — consuming zone_updates")
        try:
            async for msg in consumer:
                data = json.dumps(msg.value, default=str)
                yield f"data: {data}\n\n"
        except Exception as exc:
            log.warning("SSE consumer error: %s", exc)
        finally:
            await consumer.stop()
            log.info("SSE client disconnected")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":      "keep-alive",
        },
    )
