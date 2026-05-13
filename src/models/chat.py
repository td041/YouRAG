from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
import uuid

class ChatSession(SQLModel, table=True):
    """Bảng lưu thông tin các phiên chat"""
    __tablename__ = "chat_sessions"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    collection_name: str = Field(index=True) # Lưu trữ video/tài liệu đang chat
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Quan hệ 1-N với Messages
    messages: List["ChatMessage"] = Relationship(back_populates="session")


class ChatMessage(SQLModel, table=True):
    """Bảng lưu trữ từng dòng tin nhắn trong phiên chat"""
    __tablename__ = "chat_messages"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str = Field(foreign_key="chat_sessions.id", index=True)
    role: str = Field(description="'user' hoặc 'assistant'")
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Quan hệ ngược lại với Session
    session: ChatSession = Relationship(back_populates="messages")
