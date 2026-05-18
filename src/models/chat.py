from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, String, ForeignKey
from datetime import datetime, timezone
import uuid


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ChatSession(SQLModel, table=True):
    """Bảng lưu thông tin các phiên chat"""
    __tablename__ = "chat_sessions"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    collection_name: str = Field(index=True)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # Quan hệ 1-N với Messages (cascade delete)
    messages: List["ChatMessage"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ChatMessage(SQLModel, table=True):
    """Bảng lưu trữ từng dòng tin nhắn trong phiên chat"""
    __tablename__ = "chat_messages"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: Optional[str] = Field(
        default=None,
        sa_column=Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True),
    )
    role: str = Field(description="'user' hoặc 'assistant'")
    content: str
    created_at: datetime = Field(default_factory=_utcnow)

    # Quan hệ ngược lại với Session
    session: Optional[ChatSession] = Relationship(back_populates="messages")
