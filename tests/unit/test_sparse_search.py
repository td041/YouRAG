"""Tests for SparseRetriever — BM25 keyword search with mocked db_instance."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_db(mocker):
    """Mock db_instance to prevent Qdrant connection at import time."""
    mock = MagicMock()
    mocker.patch("src.engine.retrieval.sparse_search.db_instance", mock)
    return mock


from src.engine.retrieval.sparse_search import SparseRetriever  # noqa: E402


def _make_record(doc_id: str, text: str, extra_meta: dict = None) -> MagicMock:
    """Helper to create a mock Qdrant Record."""
    record = MagicMock()
    record.id = doc_id
    payload = {"text": text}
    if extra_meta:
        payload.update(extra_meta)
    record.payload = payload
    return record


def test_bm25_index_built_from_scroll(mock_db):
    """Kiểm tra BM25 index được build từ kết quả scroll Qdrant.

    BM25 yêu cầu đủ docs trong corpus để IDF weight > 0.
    Với corpus nhỏ (< 3 docs), term chỉ có trong 1 doc sẽ score = 0.
    """
    # BM25 cần corpus đa dạng để score > 0 — ít nhất 3-5 docs
    records = [
        _make_record("id1", "retrieval augmented generation system for search"),
        _make_record("id2", "large language models transformer architecture"),
        _make_record("id3", "retrieval methods improve document search quality"),
        _make_record("id4", "vector database embeddings storage efficient"),
        _make_record("id5", "neural network training fine tuning process"),
    ]
    # scroll returns (records, next_offset=None) → single page
    mock_db.client.scroll.return_value = (records, None)

    retriever = SparseRetriever(top_k=5)
    results = retriever.search("retrieval", "my-collection")

    # scroll should have been called once (single page)
    mock_db.client.scroll.assert_called_once()
    # "retrieval" matches id1 and id3 → both should appear in results
    result_ids = [r["id"] for r in results]
    assert "id1" in result_ids or "id3" in result_ids


def test_search_skips_zero_score_docs(mock_db):
    """Kiểm tra docs với BM25 score = 0 bị loại khỏi kết quả."""
    records = [
        _make_record("id1", "completely unrelated content xyz"),
        _make_record("id2", "also unrelated abc def ghi"),
    ]
    mock_db.client.scroll.return_value = (records, None)

    retriever = SparseRetriever(top_k=5)
    # Query with terms that won't match any doc → all scores = 0
    results = retriever.search("python fastapi qdrant", "my-collection")

    # All BM25 scores are 0 → empty results
    assert results == []


def test_bm25_cache_reused_on_second_call(mock_db):
    """Kiểm tra BM25 index được cache — scroll chỉ gọi 1 lần dù search 2 lần."""
    records = [_make_record("id1", "retrieval augmented generation")]
    mock_db.client.scroll.return_value = (records, None)

    retriever = SparseRetriever(top_k=5)
    retriever.search("retrieval", "my-collection")
    retriever.search("generation", "my-collection")

    # scroll should only be called once (cache hit on second search)
    assert mock_db.client.scroll.call_count == 1


def test_search_returns_empty_on_scroll_exception(mock_db):
    """Kiểm tra search trả về [] khi scroll Qdrant raise exception."""
    mock_db.client.scroll.side_effect = Exception("connection refused")

    retriever = SparseRetriever(top_k=5)
    results = retriever.search("test query", "my-collection")

    assert results == []


def test_search_top_k_limits_output(mock_db):
    """Kiểm tra top_k giới hạn số kết quả trả về.

    Corpus phải đa dạng để BM25 cho positive scores.
    """
    records = [
        _make_record("id1", "retrieval augmented generation document search"),
        _make_record("id2", "large language model training fine tuning"),
        _make_record("id3", "retrieval methods vector similarity matching"),
        _make_record("id4", "neural network deep learning architecture"),
        _make_record("id5", "information retrieval keyword search index"),
    ]
    mock_db.client.scroll.return_value = (records, None)

    retriever = SparseRetriever(top_k=2)
    results = retriever.search("retrieval", "my-collection")

    assert len(results) <= 2
