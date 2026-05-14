"""Unit tests for postgres.py — init_db retry logic and get_session."""

import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import OperationalError


# ── init_db tests ──────────────────────────────────────────────────────────

def test_init_db_success(mocker):
    """Kiểm tra init_db tạo bảng thành công ở lần đầu."""
    mock_create_all = mocker.patch("src.core.postgres.SQLModel.metadata.create_all")
    mocker.patch("src.core.postgres.time.sleep")

    from src.core.postgres import init_db
    init_db(retries=3, delay=1)

    mock_create_all.assert_called_once()


def test_init_db_retry_then_success(mocker):
    """Kiểm tra init_db retry khi DB chưa sẵn sàng, thành công ở lần 2."""
    mock_create_all = mocker.patch("src.core.postgres.SQLModel.metadata.create_all")
    mock_sleep = mocker.patch("src.core.postgres.time.sleep")

    # Lần 1: OperationalError, Lần 2: thành công
    mock_create_all.side_effect = [
        OperationalError("connection failed", None, None),
        None
    ]

    from src.core.postgres import init_db
    init_db(retries=3, delay=2)

    assert mock_create_all.call_count == 2
    mock_sleep.assert_called_once_with(2)


def test_init_db_raises_after_max_retries(mocker):
    """Kiểm tra init_db raise OperationalError sau khi hết retry."""
    mock_create_all = mocker.patch("src.core.postgres.SQLModel.metadata.create_all")
    mocker.patch("src.core.postgres.time.sleep")

    mock_create_all.side_effect = OperationalError("persistent failure", None, None)

    from src.core.postgres import init_db
    with pytest.raises(OperationalError):
        init_db(retries=2, delay=1)

    assert mock_create_all.call_count == 2


# ── get_session tests ──────────────────────────────────────────────────────

def test_get_session_yields_session(mocker):
    """Kiểm tra get_session trả về generator of Session."""
    mock_session = MagicMock()
    mock_session_cls = mocker.patch("src.core.postgres.Session")
    mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

    from src.core.postgres import get_session
    gen = get_session()
    session = next(gen)

    assert session is mock_session
