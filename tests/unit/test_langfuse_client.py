"""Unit tests for src/core/langfuse_client.py."""

import sys
from unittest.mock import MagicMock, patch


def _reset():
    import src.core.langfuse_client as m
    m._client = None
    m._initialized = False


def test_get_langfuse_disabled_when_keys_missing():
    """Không có LANGFUSE_PUBLIC_KEY → trả về None."""
    _reset()

    mock_settings = MagicMock()
    mock_settings.LANGFUSE_PUBLIC_KEY = None
    mock_settings.LANGFUSE_SECRET_KEY = None
    mock_settings.LANGFUSE_HOST = "http://langfuse:3000"

    fake_sdk = MagicMock()
    with patch.dict(sys.modules, {"langfuse": fake_sdk}):
        with patch("src.core.config.settings", mock_settings):
            _reset()
            from src.core.langfuse_client import get_langfuse
            result = get_langfuse()
            assert result is None

    _reset()


def test_get_langfuse_returns_none_when_sdk_missing():
    """Langfuse SDK không cài → trả về None, không raise."""
    _reset()
    with patch.dict(sys.modules, {"langfuse": None}):
        _reset()
        from src.core.langfuse_client import get_langfuse
        result = get_langfuse()
        assert result is None

    _reset()


def test_get_langfuse_returns_none_on_connect_failure(monkeypatch):
    """Langfuse SDK có nhưng kết nối lỗi → trả về None."""
    _reset()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    mock_lf_cls = MagicMock(side_effect=Exception("Connection refused"))
    fake_sdk = MagicMock()
    fake_sdk.Langfuse = mock_lf_cls

    with patch.dict(sys.modules, {"langfuse": fake_sdk}):
        _reset()
        from src.core.langfuse_client import get_langfuse
        result = get_langfuse()
        assert result is None

    _reset()


def test_get_langfuse_singleton(monkeypatch):
    """Gọi get_langfuse() lần 2 → không khởi tạo lại."""
    import src.core.langfuse_client as m
    _reset()

    sentinel = MagicMock()
    m._client = sentinel
    m._initialized = True

    from src.core.langfuse_client import get_langfuse
    result = get_langfuse()
    assert result is sentinel

    _reset()
