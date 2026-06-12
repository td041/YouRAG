"""Langfuse LLM observability client — graceful no-op when not configured."""
from typing import Optional

from src.core.logger import logger

_client: Optional[object] = None
_initialized: bool = False


def get_langfuse():
    """Return the Langfuse singleton, or None if not configured / SDK missing."""
    global _client, _initialized
    if _initialized:
        return _client

    _initialized = True
    try:
        from langfuse import Langfuse  # type: ignore[import]
        from src.core.config import settings

        public_key: Optional[str] = getattr(settings, "LANGFUSE_PUBLIC_KEY", None)
        secret_key = getattr(settings, "LANGFUSE_SECRET_KEY", None)
        host: str = getattr(settings, "LANGFUSE_HOST", "http://langfuse:3000")

        if not public_key or not secret_key:
            logger.info("[Langfuse] Keys not configured — tracing disabled")
            return None

        if hasattr(secret_key, "get_secret_value"):
            secret_key = secret_key.get_secret_value()

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            flush_at=5,
            flush_interval=2.0,
        )
        logger.info(f"✅ Langfuse tracing enabled → {host}")
    except ImportError:
        logger.warning("[Langfuse] SDK not installed (pip install 'langfuse>=2,<3') — tracing disabled")
    except Exception as e:
        logger.warning(f"[Langfuse] Failed to connect: {e} — tracing disabled")

    return _client
