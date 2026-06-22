"""Tests for ContextualCompressor — sentence-level filtering by cosine similarity."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_db(mocker):
    mock = MagicMock()
    mocker.patch("src.engine.retrieval.contextual_compressor.db_instance", mock)
    return mock


from src.engine.retrieval.contextual_compressor import (  # noqa: E402
    ContextualCompressor, _split_sentences, _cosine_sim,
)

# ── _split_sentences ──────────────────────────────────────────────────────────

def test_split_sentences_basic():
    text = (
        "Đây là câu đầu tiên trong đoạn văn này. "
        "Câu thứ hai cũng đủ dài để không bị bỏ. "
        "Và đây là câu thứ ba hoàn chỉnh nhất."
    )
    parts = _split_sentences(text)
    assert len(parts) == 3


def test_split_sentences_drops_short():
    text = "OK. Đây là một câu đủ dài để không bị bỏ qua trong bài kiểm tra này."
    parts = _split_sentences(text)
    assert all(len(p) >= 20 for p in parts)


# ── _cosine_sim ───────────────────────────────────────────────────────────────

def test_cosine_sim_identical():
    v = [1.0, 0.0, 0.0]
    assert _cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_sim_orthogonal():
    assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_sim_zero_vector():
    assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


# ── ContextualCompressor.compress ─────────────────────────────────────────────

def _make_chunk(content: str) -> dict:
    return {"id": "x", "content": content, "metadata": {}}

LONG_SENTENCE_A = "Đây là câu đầu tiên liên quan đến chủ đề chính của tài liệu này."
LONG_SENTENCE_B = "Câu thứ hai cũng rất liên quan và mô tả nội dung quan trọng của video."
LONG_SENTENCE_C = "Câu này hoàn toàn không liên quan gì đến truy vấn của người dùng cả."


def test_compress_empty_returns_empty(mock_db):
    mock_db.embedding_model.encode.return_value = [[0.1, 0.2]]
    compressor = ContextualCompressor()
    assert compressor.compress("query", []) == []


def test_compress_short_chunk_kept_unchanged(mock_db):
    """Chunk dưới 3 câu phải giữ nguyên."""
    mock_db.embedding_model.encode.return_value = [[0.5, 0.5]]
    compressor = ContextualCompressor(threshold=0.9)
    chunk = _make_chunk("Chỉ có một câu duy nhất ngắn thôi.")
    result = compressor.compress("query", [chunk])
    assert result[0]["content"] == chunk["content"]


def test_compress_filters_irrelevant_sentences(mock_db):
    """Câu có similarity thấp phải bị lọc bỏ."""
    import numpy as np
    query_vec = np.array([1.0, 0.0])
    sent_vecs = [
        np.array([0.9, 0.1]),   # relevant
        np.array([0.85, 0.15]), # relevant
        np.array([0.0, 1.0]),   # irrelevant
    ]
    mock_db.embedding_model.encode.side_effect = [[query_vec], sent_vecs]
    compressor = ContextualCompressor(threshold=0.6)
    chunk = _make_chunk(f"{LONG_SENTENCE_A} {LONG_SENTENCE_B} {LONG_SENTENCE_C}")
    result = compressor.compress("query", [chunk])
    assert LONG_SENTENCE_C not in result[0]["content"]
    assert LONG_SENTENCE_A in result[0]["content"] or LONG_SENTENCE_B in result[0]["content"]


def test_compress_fallback_keeps_all_if_all_filtered(mock_db):
    """Nếu lọc hết thì giữ nguyên chunk gốc."""
    import numpy as np
    query_vec = np.array([1.0, 0.0])
    sent_vecs = [
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    ]
    mock_db.embedding_model.encode.side_effect = [[query_vec], sent_vecs]
    compressor = ContextualCompressor(threshold=0.8)
    original = f"{LONG_SENTENCE_A} {LONG_SENTENCE_B} {LONG_SENTENCE_C}"
    result = compressor.compress("query", [_make_chunk(original)])
    assert result[0]["content"] == original


def test_compress_encode_failure_returns_original(mock_db):
    """Lỗi khi encode query → trả về chunks gốc."""
    mock_db.embedding_model.encode.side_effect = RuntimeError("CUDA OOM")
    compressor = ContextualCompressor()
    chunk = _make_chunk("Nội dung bất kỳ của chunk này để test fallback.")
    result = compressor.compress("query", [chunk])
    assert result == [chunk]
