"""Unit tests for FastAPI endpoints in src/api/main.py."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Mock các model ngay lập tức để tránh load model thật khi import app
with patch("src.engine.ingestion.pipeline.IngestionPipeline"), \
     patch("src.engine.ranking.cross_encoder.CrossEncoderReranker"), \
     patch("src.engine.generation.answer_generator.AnswerGenerator"), \
     patch("src.engine.generation.summarizer.VideoSummarizer"), \
     patch("src.engine.retrieval.graph_rag.GraphRetriever"), \
     patch("src.cache.semantic_cache.SemanticCache"), \
     patch("src.core.postgres.init_db"):
    from src.api.main import app, AIStore
    from src.api.auth import require_api_key

# Override auth để không cần API key trong tests
app.dependency_overrides[require_api_key] = lambda: ""

client = TestClient(app)


@pytest.fixture
def mock_aistore():
    """Mock các component trong AIStore."""
    AIStore.pipeline = MagicMock()
    AIStore.reranker = MagicMock()
    AIStore.generator = MagicMock()
    AIStore.summarizer = MagicMock()
    AIStore.graph_retriever = MagicMock()
    AIStore.cache = MagicMock()
    return AIStore


def test_read_root():
    """Kiểm tra endpoint root."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_list_collections(mocker):
    """Kiểm tra endpoint list collections."""
    mock_db = mocker.patch("src.api.main.db_instance")
    
    # Mock Qdrant collection response
    mock_col = MagicMock()
    mock_col.name = "test_col"
    mock_db.client.get_collections.return_value.collections = [mock_col]
    
    # Mock scroll return
    mock_rec = MagicMock()
    mock_rec.payload = {"title": "Test Video", "video_id": "123"}
    mock_db.client.scroll.return_value = ([mock_rec], None)
    
    response = client.get("/collections")
    assert response.status_code == 200
    assert response.json()[0]["title"] == "Test Video"


def test_ingest_video(mock_aistore):
    """Kiểm tra endpoint ingest trả về job_id (async)."""
    mock_aistore.pipeline.run.return_value = {"status": "success", "chunks_added": 10}

    response = client.post("/ingest", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_chat_rag_cache_hit(mock_aistore):
    """Kiểm tra endpoint chat khi có cache hit."""
    mock_aistore.cache.check_cache.return_value = {
        "answer": "Cached answer",
        "sources": ["source1"],
        "facts": ["fact1"]
    }
    
    response = client.post("/chat", json={
        "query": "hello",
        "collection": "test_col"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Cached answer"
    assert data["cached"] is True
    # check_cache được gọi với collection_name
    mock_aistore.cache.check_cache.assert_called_once_with("hello", collection_name="test_col")
    # LLM generator không được gọi
    mock_aistore.generator.generate.assert_not_called()


def test_chat_rag_cache_miss(mock_aistore, mocker):
    """Kiểm tra endpoint chat khi cache miss và phải gọi RAG pipeline."""
    mock_aistore.cache.check_cache.return_value = None
    mock_aistore.generator.generate.return_value = "AI answer"
    mock_aistore.graph_retriever.search.return_value = {"facts": [], "entities": []}
    mock_aistore.reranker.rerank.return_value = [{"metadata": {"start_time": 0, "end_time": 10}}]
    
    # Mock HybridRetriever
    mocker.patch("src.api.main.HybridRetriever")
    
    response = client.post("/chat", json={
        "query": "hello",
        "collection": "test_col"
    })
    
    assert response.status_code == 200
    assert response.json()["answer"] == "AI answer"
    assert response.json()["cached"] is False
    mock_aistore.cache.save_to_cache.assert_called_once()


def test_chat_rag_stream_cache_hit(mock_aistore):
    """Kiểm tra endpoint chat stream khi có cache hit."""
    mock_aistore.cache.check_cache.return_value = {
        "answer": "Cached answer",
        "sources": ["source1"],
        "facts": ["fact1"]
    }
    
    response = client.post("/chat/stream", json={
        "query": "hello",
        "collection": "test_col"
    })
    
    assert response.status_code == 200
    assert "Cached answer" in response.text
    # New JSON meta frame format
    assert "__META__" in response.text
    assert '"cached": true' in response.text or '"cached":true' in response.text


def test_get_summary(mock_aistore):
    """Kiểm tra endpoint summarize."""
    mock_aistore.summarizer.summarize.return_value = "Video summary"
    
    response = client.get("/summarize/test_col")
    assert response.status_code == 200
    assert response.json()["summary"] == "Video summary"


def test_build_graph_endpoint(mock_aistore, mocker):
    """Kiểm tra endpoint build graph."""
    mock_builder_cls = mocker.patch("src.api.main.KnowledgeGraphBuilder")
    mock_builder = MagicMock()
    mock_g = MagicMock()
    mock_g.number_of_nodes.return_value = 10
    mock_g.number_of_edges.return_value = 20
    mock_builder.build_graph.return_value = mock_g
    mock_builder_cls.return_value = mock_builder
    
    # Setup graph_retriever cache mock
    mock_aistore.graph_retriever._graph_cache = {}
    
    response = client.post("/graph/build/test_col")
    assert response.status_code == 200
    assert response.json()["nodes"] == 10


def test_get_chat_history_endpoint(mocker):
    """Kiểm tra endpoint lấy lịch sử chat."""
    mock_hist_cls = mocker.patch("src.api.main.ChatHistoryManager")
    mock_hist = MagicMock()
    mock_hist.get_history.return_value = [{"role": "user", "content": "hi"}]
    mock_hist_cls.return_value = mock_hist
    
    response = client.get("/history/session123?collection=test_col")
    assert response.status_code == 200
    assert response.json()[0]["content"] == "hi"


def test_summarize_stream_endpoint(mock_aistore):
    """Kiểm tra endpoint summarize stream."""
    mock_aistore.summarizer.summarize_stream.return_value = iter(["Summary", " chunk"])

    response = client.get("/summarize/stream/test_col")
    assert response.status_code == 200
    assert "Summary chunk" in response.text


# ─────────────────────────────────────────────
# AUTH TESTS
# ─────────────────────────────────────────────

def test_auth_no_key_when_not_configured():
    """Khi API_KEY chưa cấu hình → dev mode, không cần key."""
    from src.api import auth as auth_module
    # Simulate no key configured
    with patch.object(auth_module.settings, "API_KEY", None):
        from src.api.auth import require_api_key as _rk
        result = _rk(key=None)
        assert result == ""


def test_auth_valid_key():
    """Key đúng → trả về key."""
    from unittest.mock import MagicMock
    from src.api import auth as auth_module
    mock_secret = MagicMock()
    mock_secret.get_secret_value.return_value = "secret123"
    with patch.object(auth_module.settings, "API_KEY", mock_secret):
        from src.api.auth import require_api_key as _rk
        result = _rk(key="secret123")
        assert result == "secret123"


def test_auth_invalid_key():
    """Key sai → 401."""
    from fastapi import HTTPException
    from unittest.mock import MagicMock
    from src.api import auth as auth_module
    mock_secret = MagicMock()
    mock_secret.get_secret_value.return_value = "secret123"
    with patch.object(auth_module.settings, "API_KEY", mock_secret):
        from src.api.auth import require_api_key as _rk
        with pytest.raises(HTTPException) as exc_info:
            _rk(key="wrong")
        assert exc_info.value.status_code == 401


def test_protected_endpoint_returns_401_without_key():
    """Endpoint được bảo vệ → 401 khi thiếu key (restore override)."""
    from unittest.mock import MagicMock
    from src.api import auth as auth_module
    mock_secret = MagicMock()
    mock_secret.get_secret_value.return_value = "secret123"

    # Remove override tạm thời để test thật
    app.dependency_overrides.pop(require_api_key, None)
    try:
        with patch.object(auth_module.settings, "API_KEY", mock_secret):
            response = client.get("/summarize/test_col")
            assert response.status_code == 401
    finally:
        app.dependency_overrides[require_api_key] = lambda: ""


def test_protected_endpoint_returns_200_with_valid_key(mock_aistore):
    """Endpoint được bảo vệ → 200 khi có đúng key."""
    from unittest.mock import MagicMock
    from src.api import auth as auth_module
    mock_secret = MagicMock()
    mock_secret.get_secret_value.return_value = "secret123"
    mock_aistore.summarizer.summarize.return_value = "summary"

    app.dependency_overrides.pop(require_api_key, None)
    try:
        with patch.object(auth_module.settings, "API_KEY", mock_secret):
            response = client.get("/summarize/test_col", headers={"X-API-Key": "secret123"})
            assert response.status_code == 200
    finally:
        app.dependency_overrides[require_api_key] = lambda: ""
