import pytest
from unittest.mock import MagicMock
from sqlmodel import SQLModel, create_engine, Session
from src.models.chat import ChatSession, ChatMessage

# Import AFTER creating the mock engine if possible, or patch it
# Since src.engine.chat.history imports engine from src.core.postgres, we must patch it there.

@pytest.fixture(name="db_session")
def db_session_fixture():
    """Tạo DB SQLite in-memory để test."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture(autouse=True)
def mock_deps(mocker):
    """Mock các phụ thuộc nặng hoặc bên ngoài."""
    # 1. Mock Redis Client trực tiếp trong module history
    mock_redis = MagicMock()
    mocker.patch("src.engine.chat.history.redis_client", mock_redis)
    
    # 2. Mock Postgres Engine trực tiếp trong module history
    test_engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(test_engine)
    mocker.patch("src.engine.chat.history.engine", test_engine)
    
    return mock_redis, test_engine

def test_session_creation(mock_deps):
    """Kiểm tra xem ChatSession có được tạo tự động nếu chưa tồn tại không."""
    mock_redis, test_engine = mock_deps
    from src.engine.chat.history import ChatHistoryManager
    
    session_id = "test_session_123"
    collection = "test_video"
    
    # Khởi tạo manager
    ChatHistoryManager(session_id=session_id, collection_name=collection)
    
    # Kiểm tra trong DB
    with Session(test_engine) as db:
        session = db.get(ChatSession, session_id)
        assert session is not None
        assert session.collection_name == collection

def test_add_message(mock_deps):
    """Kiểm tra thêm tin nhắn vào DB và Redis."""
    mock_redis, test_engine = mock_deps
    from src.engine.chat.history import ChatHistoryManager
    
    session_id = "test_session_456"
    manager = ChatHistoryManager(session_id=session_id, collection_name="test")
    
    # Thêm tin nhắn
    manager.add_message(role="user", content="Hello AI")
    
    # 1. Kiểm tra DB
    with Session(test_engine) as db:
        from sqlmodel import select
        statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
        msg = db.exec(statement).first()
        assert msg is not None
        assert msg.role == "user"
        assert msg.content == "Hello AI"
        
        # Kiểm tra updated_at của session
        session = db.get(ChatSession, session_id)
        assert session.updated_at is not None

    # 2. Kiểm tra Redis
    assert mock_redis.rpush.called
    assert mock_redis.expire.called

def test_get_history_from_redis(mock_deps):
    """Kiểm tra lấy lịch sử từ Redis (ưu tiên cao hơn)."""
    mock_redis, test_engine = mock_deps
    from src.engine.chat.history import ChatHistoryManager
    import json
    
    session_id = "test_session_redis"
    manager = ChatHistoryManager(session_id=session_id, collection_name="test")
    
    # Giả lập Redis có dữ liệu
    mock_redis.exists.return_value = True
    mock_redis.lrange.return_value = [
        json.dumps({"role": "user", "content": "Hi"}),
        json.dumps({"role": "assistant", "content": "Hello!"})
    ]
    
    history = manager.get_history(limit=5)
    
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hello!"
    # Đảm bảo không gọi xuống Postgres (trong thực tế nó sẽ không gọi nếu có Redis)
    # Ở đây chúng ta chỉ kiểm tra kết quả trả về đúng từ Redis mock

def test_get_history_fallback_to_db(mock_deps):
    """Kiểm tra fallback xuống DB khi Redis trống."""
    mock_redis, test_engine = mock_deps
    from src.engine.chat.history import ChatHistoryManager
    
    session_id = "test_session_fallback"
    manager = ChatHistoryManager(session_id=session_id, collection_name="test")
    
    # Redis báo không tồn tại
    mock_redis.exists.return_value = False
    
    # Thêm dữ liệu trực tiếp vào DB
    with Session(test_engine) as db:
        msg1 = ChatMessage(session_id=session_id, role="user", content="Question 1")
        msg2 = ChatMessage(session_id=session_id, role="assistant", content="Answer 1")
        db.add(msg1)
        db.add(msg2)
        db.commit()
    
    history = manager.get_history(limit=5)
    
    assert len(history) == 2
    assert history[0]["content"] == "Question 1"
    assert history[1]["content"] == "Answer 1"
    
    # Kiểm tra xem có cố gắng phục hồi (backfill) vào Redis không
    assert mock_redis.pipeline.called

def test_format_for_prompt(mock_deps):
    """Kiểm tra định dạng chuỗi cho Prompt."""
    mock_redis, test_engine = mock_deps
    from src.engine.chat.history import ChatHistoryManager
    import json
    
    session_id = "test_session_format"
    manager = ChatHistoryManager(session_id=session_id, collection_name="test")
    
    mock_redis.exists.return_value = True
    mock_redis.lrange.return_value = [
        json.dumps({"role": "user", "content": "A"}),
        json.dumps({"role": "assistant", "content": "B"})
    ]
    
    formatted = manager.format_for_prompt(limit=2)
    
    assert "Người dùng: A" in formatted
    assert "AI: B" in formatted
    assert "--- Lịch sử trò chuyện trước đó ---" in formatted
