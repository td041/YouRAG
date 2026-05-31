"""Redis client singleton — shared across graph store and contextual cache."""
import redis
from src.core.config import settings
from src.core.logger import logger

_client: redis.Redis | None = None


def get_redis() -> redis.Redis | None:
    """Return Redis client, or None if unavailable (dev fallback)."""
    global _client
    if _client is not None:
        return _client
    try:
        r = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        _client = r
        logger.info("✅ Redis client connected")
        return _client
    except Exception as e:
        logger.warning(f"⚠️  Redis unavailable ({e}) — falling back to local storage")
        return None
