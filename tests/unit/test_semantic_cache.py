"""Tests for SemanticCache — Qdrant-backed semantic cache with mocked db_instance."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_db(mocker):
    """Mock db_instance and settings to prevent real Qdrant/SentenceTransformer I/O."""
    mock = MagicMock()
    # encode() returns a list whose first element has .tolist()
    fake_vec = MagicMock()
    fake_vec.tolist.return_value = [0.1, 0.2, 0.3]
    mock.embedding_model.encode.return_value = [fake_vec]

    mocker.patch("src.cache.semantic_cache.db_instance", mock)
    mocker.patch("src.cache.semantic_cache.settings", MagicMock(SEMANTIC_CACHE_THRESHOLD=0.92))
    return mock


from src.cache.semantic_cache import SemanticCache  # noqa: E402


# ─── Initialization ───────────────────────────────────────────────────────────

def test_init_creates_collection(mock_db):
    """Kiểm tra __init__ gọi get_or_create_collection với đúng tên collection."""
    SemanticCache()
    mock_db.get_or_create_collection.assert_called_once_with("semantic_cache")


def test_init_default_threshold(mock_db):
    """Kiểm tra threshold mặc định lấy từ settings khi không truyền tham số."""
    cache = SemanticCache()
    assert cache.threshold == 0.92


def test_init_custom_threshold(mock_db):
    """Kiểm tra threshold tùy chỉnh ghi đè giá trị từ settings."""
    cache = SemanticCache(similarity_threshold=0.85)
    assert cache.threshold == 0.85


# ─── check_cache ──────────────────────────────────────────────────────────────

def test_check_cache_hit_returns_answer(mock_db):
    """Kiểm tra check_cache trả về answer khi tìm thấy điểm có score >= threshold."""
    fake_point = MagicMock()
    fake_point.score = 0.95
    fake_point.payload = {"query": "what is AI?", "answer": "AI is artificial intelligence."}
    mock_db.client.query_points.return_value.points = [fake_point]

    cache = SemanticCache()
    result = cache.check_cache("what is AI?")

    assert result == "AI is artificial intelligence."
    mock_db.client.query_points.assert_called_once()


def test_check_cache_miss_returns_none(mock_db):
    """Kiểm tra check_cache trả về None khi không tìm thấy kết quả nào đủ điểm."""
    mock_db.client.query_points.return_value.points = []

    cache = SemanticCache()
    result = cache.check_cache("unknown query")

    assert result is None


def test_check_cache_passes_score_threshold(mock_db):
    """Kiểm tra check_cache truyền đúng score_threshold vào query_points."""
    mock_db.client.query_points.return_value.points = []

    cache = SemanticCache(similarity_threshold=0.88)
    cache.check_cache("some query")

    call_kwargs = mock_db.client.query_points.call_args.kwargs
    assert call_kwargs["score_threshold"] == 0.88
    assert call_kwargs["collection_name"] == "semantic_cache"
    assert call_kwargs["limit"] == 1


def test_check_cache_returns_none_on_exception(mock_db):
    """Kiểm tra check_cache trả về None (không raise) khi Qdrant gặp lỗi."""
    mock_db.client.query_points.side_effect = Exception("Qdrant unavailable")

    cache = SemanticCache()
    result = cache.check_cache("any query")

    assert result is None


# ─── save_to_cache ────────────────────────────────────────────────────────────

def test_save_to_cache_calls_upsert(mock_db):
    """Kiểm tra save_to_cache gọi client.upsert với payload chứa query và answer."""
    cache = SemanticCache()
    cache.save_to_cache("what is ML?", "ML is machine learning.")

    mock_db.client.upsert.assert_called_once()
    call_kwargs = mock_db.client.upsert.call_args.kwargs
    assert call_kwargs["collection_name"] == "semantic_cache"

    point = call_kwargs["points"][0]
    assert point.payload["query"] == "what is ML?"
    assert point.payload["answer"] == "ML is machine learning."
    assert point.vector == [0.1, 0.2, 0.3]


def test_save_to_cache_deterministic_id(mock_db):
    """Kiểm tra save_to_cache tạo cùng ID cho cùng một query (idempotent upsert)."""
    cache = SemanticCache()
    cache.save_to_cache("hello", "world")
    id1 = mock_db.client.upsert.call_args.kwargs["points"][0].id

    mock_db.client.upsert.reset_mock()
    cache.save_to_cache("hello", "world")
    id2 = mock_db.client.upsert.call_args.kwargs["points"][0].id

    assert id1 == id2


def test_save_to_cache_different_queries_different_ids(mock_db):
    """Kiểm tra hai query khác nhau tạo ra ID khác nhau."""
    cache = SemanticCache()
    cache.save_to_cache("query A", "answer A")
    id_a = mock_db.client.upsert.call_args.kwargs["points"][0].id

    mock_db.client.upsert.reset_mock()
    cache.save_to_cache("query B", "answer B")
    id_b = mock_db.client.upsert.call_args.kwargs["points"][0].id

    assert id_a != id_b


def test_save_to_cache_silent_on_exception(mock_db):
    """Kiểm tra save_to_cache không raise khi Qdrant gặp lỗi (graceful degradation)."""
    mock_db.client.upsert.side_effect = Exception("disk full")

    cache = SemanticCache()
    # Must not raise
    cache.save_to_cache("query", "answer")


# ─── Integration: check then save roundtrip ───────────────────────────────────

def test_roundtrip_miss_then_hit(mock_db):
    """Kiểm tra quy trình đầy đủ: miss → save → hit trả về đúng answer."""
    # First call: miss
    mock_db.client.query_points.return_value.points = []
    cache = SemanticCache()
    assert cache.check_cache("What is deep learning?") is None

    # Simulate save
    cache.save_to_cache("What is deep learning?", "Deep learning is a subset of ML.")

    # Second call: hit
    fake_point = MagicMock()
    fake_point.score = 0.97
    fake_point.payload = {"query": "What is deep learning?", "answer": "Deep learning is a subset of ML."}
    mock_db.client.query_points.return_value.points = [fake_point]

    result = cache.check_cache("What is deep learning?")
    assert result == "Deep learning is a subset of ML."
