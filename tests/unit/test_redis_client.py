"""Unit tests for src/core/redis_client.py."""

from unittest.mock import MagicMock


def test_get_redis_returns_client_on_success(mocker):
    """Kết nối Redis thành công → trả về client."""
    import src.core.redis_client as rc_module

    rc_module._client = None  # reset singleton

    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    mocker.patch("redis.from_url", return_value=mock_redis)

    from src.core.redis_client import get_redis

    result = get_redis()
    assert result is mock_redis


def test_get_redis_returns_none_on_connection_error(mocker):
    """Redis không kết nối được → trả về None, không raise exception."""
    import src.core.redis_client as rc_module

    rc_module._client = None

    mocker.patch("redis.from_url", side_effect=Exception("Connection refused"))

    from src.core.redis_client import get_redis

    result = get_redis()
    assert result is None


def test_get_redis_reuses_singleton(mocker):
    """Lần thứ hai gọi get_redis() → không gọi from_url lại."""
    import src.core.redis_client as rc_module

    mock_client = MagicMock()
    rc_module._client = mock_client  # pre-seed singleton

    mock_from_url = mocker.patch("redis.from_url")

    from src.core.redis_client import get_redis

    result = get_redis()
    assert result is mock_client
    mock_from_url.assert_not_called()

    # cleanup
    rc_module._client = None
