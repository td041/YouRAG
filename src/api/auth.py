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
    # Dev mode: API_KEY chưa set hoặc set thành chuỗi rỗng → bỏ qua kiểm tra
    if configured is None or configured.get_secret_value().strip() == "":
        return ""
    if key != configured.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide X-API-Key header.",
        )
    return key
