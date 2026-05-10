"""Tests for DenseRetriever — Qdrant vector search with mocked db_instance."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_db(mocker):
    """Mock db_instance to prevent loading SentenceTransformer and Qdrant at import time."""
    mock = MagicMock()
    mocker.patch("src.engine.retrieval.dense_search.db_instance", mock)
    return mock


from src.engine.retrieval.dense_search import DenseRetriever  # noqa: E402


def test_init_stores_top_k():
    """Kiểm tra khởi tạo lưu đúng giá trị top_k."""
    retriever = DenseRetriever(top_k=7)
    assert retriever.top_k == 7


def test_search_returns_formatted_chunks(mock_db):
    """Kiểm tra search trả về chunks được định dạng đúng từ Qdrant response."""
    # Simulate Qdrant ScoredPoint
    fake_point = MagicMock()
    fake_point.id = "abc-123"
    fake_point.score = 0.92
    fake_point.payload = {"text": "Hello world", "start_time": 10.0, "chunk_index": 0}

    mock_db.embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
    mock_db.client.search.return_value = [fake_point]

    retriever = DenseRetriever(top_k=5)
    results = retriever.search("test query", "my-collection")

    assert len(results) == 1
    assert results[0]["id"] == "abc-123"
    assert results[0]["content"] == "Hello world"
    assert results[0]["distance"] == 0.92
    assert results[0]["metadata"]["start_time"] == 10.0
    # "text" key should be popped from metadata
    assert "text" not in results[0]["metadata"]


def test_search_returns_empty_on_no_results(mock_db):
    """Kiểm tra search trả về [] khi Qdrant không có kết quả."""
    mock_db.embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
    mock_db.client.search.return_value = []

    retriever = DenseRetriever(top_k=5)
    results = retriever.search("test query", "empty-collection")

    assert results == []


def test_search_returns_empty_on_exception(mock_db):
    """Kiểm tra search trả về [] khi Qdrant raise exception (graceful degradation)."""
    mock_db.embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
    mock_db.client.search.side_effect = Exception("connection refused")

    retriever = DenseRetriever(top_k=5)
    results = retriever.search("test query", "my-collection")

    assert results == []
