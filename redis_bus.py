"""Optional Redis-backed message queue wrapper."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None


class RedisMessageBus:
    """Tiny Redis queue abstraction used for future asynchronous agent messages."""

    def __init__(self, url: str = "redis://localhost:6379/0", queue_name: str = "mapfm:agent_messages") -> None:
        self.url = url
        self.queue_name = queue_name
        self.client = redis.Redis.from_url(url, decode_responses=True) if redis is not None else None

    def publish(self, payload: Dict[str, Any]) -> bool:
        """Push a JSON payload to the configured Redis queue."""
        if self.client is None:
            return False
        self.client.rpush(self.queue_name, json.dumps(payload, ensure_ascii=False, default=str))
        return True

    def consume(self, timeout_seconds: int = 1) -> Optional[Dict[str, Any]]:
        """Pop a queue message with a bounded wait."""
        if self.client is None:
            return None
        item = self.client.blpop(self.queue_name, timeout=timeout_seconds)
        if item is None:
            return None
        _, raw = item
        return json.loads(raw)
