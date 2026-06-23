"""Unit tests for src/models/chat.py — SQLModel table definitions."""

from datetime import datetime, timezone


def test_chat_session_default_fields():
    """ChatSession có id UUID và created_at mặc định."""
    from src.models.chat import ChatSession

    session = ChatSession(collection_name="test-collection")
    assert session.id is not None
    assert len(session.id) == 36  # UUID format
    assert session.collection_name == "test-collection"
    assert isinstance(session.created_at, datetime)


def test_chat_message_default_fields():
    """ChatMessage có id UUID và role/content bắt buộc."""
    from src.models.chat import ChatMessage

    msg = ChatMessage(role="user", content="Hello")
    assert msg.id is not None
    assert len(msg.id) == 36
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert isinstance(msg.created_at, datetime)


def test_chat_message_roles():
    """ChatMessage chấp nhận role 'user' và 'assistant'."""
    from src.models.chat import ChatMessage

    user_msg = ChatMessage(role="user", content="question")
    assistant_msg = ChatMessage(role="assistant", content="answer")

    assert user_msg.role == "user"
    assert assistant_msg.role == "assistant"


def test_utcnow_is_timezone_aware():
    """_utcnow() trả về datetime timezone-aware (UTC)."""
    from src.models.chat import _utcnow

    dt = _utcnow()
    assert dt.tzinfo is not None
    assert dt.tzinfo == timezone.utc


def test_chat_session_unique_ids():
    """Mỗi ChatSession có UUID khác nhau."""
    from src.models.chat import ChatSession

    s1 = ChatSession(collection_name="col1")
    s2 = ChatSession(collection_name="col2")
    assert s1.id != s2.id
