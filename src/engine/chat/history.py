import json
import redis
from typing import List, Dict
from sqlmodel import Session, select
from datetime import datetime

from src.core.config import settings
from src.core.postgres import engine
from src.models.chat import ChatSession, ChatMessage
from src.core.logger import setup_logger

logger = setup_logger("ChatHistory")

# Cấu hình Redis từ centralized settings
REDIS_URL = settings.REDIS_URL
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
CACHE_TTL = 3600 * 24  # 1 ngày lưu trên Redis

class ChatHistoryManager:
    """Quản lý lịch sử chat: Tốc độ cao với Redis, Bền vững với PostgreSQL"""

    def __init__(self, session_id: str, collection_name: str):
        self.session_id = session_id
        self.collection_name = collection_name
        self.redis_key = f"chat_history:{self.session_id}"
        self._ensure_session_exists()

    def _ensure_session_exists(self):
        """Đảm bảo Session tồn tại trong PostgreSQL"""
        try:
            with Session(engine) as db:
                session = db.get(ChatSession, self.session_id)
                if not session:
                    new_session = ChatSession(id=self.session_id, collection_name=self.collection_name)
                    db.add(new_session)
                    db.commit()
                    logger.info(f"Đã tạo ChatSession mới trong DB: {self.session_id}")
        except Exception as e:
            logger.error(f"❌ Lỗi khi khởi tạo ChatSession trong Postgres: {e}")

    def add_message(self, role: str, content: str):
        """Lưu tin nhắn vào cả Redis và PostgreSQL"""
        # 1. Lưu vào PostgreSQL
        try:
            with Session(engine) as db:
                msg = ChatMessage(session_id=self.session_id, role=role, content=content)
                db.add(msg)
                # Cập nhật thời gian session
                session = db.get(ChatSession, self.session_id)
                if session:
                    session.updated_at = datetime.utcnow()
                    db.add(session)
                db.commit()
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu tin nhắn vào Postgres: {e}")

        # 2. Lưu vào Redis (với try-except để không crash nếu Redis die)
        try:
            msg_data = {"role": role, "content": content}
            redis_client.rpush(self.redis_key, json.dumps(msg_data))
            redis_client.expire(self.redis_key, CACHE_TTL)
        except redis.exceptions.RedisError as e:
            logger.warning(f"⚠️ Redis đang gặp sự cố, bỏ qua lưu cache: {e}")

    def get_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Lấy lịch sử chat. Ưu tiên Redis, nếu không có thì lấy từ PostgreSQL."""
        
        # 1. Thử lấy từ Redis
        try:
            if redis_client.exists(self.redis_key):
                # Lấy K tin nhắn cuối cùng
                raw_msgs = redis_client.lrange(self.redis_key, -limit, -1)
                if raw_msgs:
                    logger.info(f"[{self.session_id}] Lấy lịch sử từ REDIS")
                    return [json.loads(m) for m in raw_msgs]
        except redis.exceptions.RedisError as e:
            logger.warning(f"⚠️ Lỗi kết nối Redis ({e}), đang fallback xuống Postgres...")
        
        # 2. Fallback: Lấy từ PostgreSQL nếu Redis mất (cache miss hoặc error)
        logger.info(f"[{self.session_id}] Lấy lịch sử từ POSTGRES")
        try:
            with Session(engine) as db:
                statement = select(ChatMessage).where(ChatMessage.session_id == self.session_id).order_by(ChatMessage.created_at.desc()).limit(limit)
                db_msgs = db.exec(statement).all()
                
                # Đảo ngược lại để đúng thứ tự thời gian (cũ -> mới)
                db_msgs.reverse()
                
                history = [{"role": msg.role, "content": msg.content} for msg in db_msgs]
                
                # Thử phục hồi vào Redis nếu có thể
                if history:
                    try:
                        pipe = redis_client.pipeline()
                        for msg in history:
                            pipe.rpush(self.redis_key, json.dumps(msg))
                        pipe.expire(self.redis_key, CACHE_TTL)
                        pipe.execute()
                    except redis.exceptions.RedisError:
                        pass # Bỏ qua nếu phục hồi cache thất bại

                return history
        except Exception as e:
            logger.error(f"❌ Lỗi khi truy vấn Postgres: {e}")
            return []

    def format_for_prompt(self, limit: int = 4) -> str:
        """Định dạng lịch sử thành chuỗi văn bản để nhét vào System Prompt"""
        history = self.get_history(limit=limit)
        if not history:
            return ""
        
        formatted = "--- Lịch sử trò chuyện trước đó ---\n"
        for msg in history:
            role_name = "Người dùng" if msg["role"] == "user" else "AI"
            formatted += f"{role_name}: {msg['content']}\n\n"
        formatted += "-----------------------------------\n"
        return formatted
