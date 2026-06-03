"""API key authentication dependency for YouRAG endpoints."""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.core.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(_api_key_header)) -> str:
    """FastAPI dependency — bảo vệ endpoint bằng API key.

    Nếu API_KEY không được cấu hình trong settings → bỏ qua kiểm tra (dev mode).
    Nếu đã cấu hình → yêu cầu header X-API-Key khớp.
    """
    configured = settings.API_KEY
    # Dev mode: API_KEY not set at all → skip auth
    if configured is None:
        return ""
    secret = configured.get_secret_value()
    # Empty string from Docker env (API_KEY=) → dev mode, not misconfigured
    if secret == "":
        return ""
    # Whitespace-only key is a misconfiguration — reject all requests
    if not secret.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server misconfigured: API_KEY contains only whitespace.",
        )
    if key != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )
    return key
