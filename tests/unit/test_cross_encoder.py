"""Unit tests for CrossEncoderReranker."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_cross_encoder_deps(mocker):
    """Mock sentence_transformers.CrossEncoder to avoid loading a real model."""
    mock_cls = mocker.patch("src.engine.ranking.cross_encoder.CrossEncoder")
    mock_instance = MagicMock()
    # Mock predict to return scores for the number of pairs passed
    mock_instance.predict.side_effect = lambda pairs: [0.9, 0.1, 0.5][:len(pairs)]
    mock_cls.return_value = mock_instance
    return mock_cls, mock_instance


from src.engine.ranking.cross_encoder import CrossEncoderReranker  # noqa: E402


def test_init_loads_model(mock_cross_encoder_deps):
    """Kiểm tra khởi tạo có load đúng model name."""
    mock_cls, _ = mock_cross_encoder_deps
    reranker = CrossEncoderReranker()
    assert reranker.model_name is not None
    mock_cls.assert_called_once()


def test_rerank_empty_chunks():
    """Kiểm tra rerank trả về list rỗng khi đầu vào rỗng."""
    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", [])
    assert result == []


def test_rerank_fallback_when_model_none(mocker):
    """Kiểm tra fallback trả về top_k chunks gốc khi model load lỗi."""
    mocker.patch("src.engine.ranking.cross_encoder.CrossEncoder", side_effect=Exception("Load error"))
    reranker = CrossEncoderReranker()
    assert reranker._model is None
    
    chunks = [{"content": "A"}, {"content": "B"}]
    result = reranker.rerank("query", chunks, top_k=1)
    assert result == [{"content": "A"}]


def test_rerank_reorders_chunks(mock_cross_encoder_deps):
    """Kiểm tra rerank sắp xếp lại chunks dựa trên score từ model."""
    _, mock_instance = mock_cross_encoder_deps
    # Scores: 0.9, 0.1, 0.5 for A, B, C
    mock_instance.predict.return_value = [0.9, 0.1, 0.5]
    
    chunks = [
        {"content": "A"}, # score 0.9
        {"content": "B"}, # score 0.1
        {"content": "C"}  # score 0.5
    ]
    
    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", chunks, top_k=3)
    
    # Expected order: A (0.9), C (0.5), B (0.1)
    assert result[0]["content"] == "A"
    assert result[1]["content"] == "C"
    assert result[2]["content"] == "B"
    assert "rerank_score" in result[0]


def test_rerank_handles_exception_gracefully(mock_cross_encoder_deps):
    """Kiểm tra rerank trả về chunks gốc nếu quá trình predict bị lỗi."""
    _, mock_instance = mock_cross_encoder_deps
    mock_instance.predict.side_effect = Exception("Prediction error")
    
    reranker = CrossEncoderReranker()
    chunks = [{"content": "A"}, {"content": "B"}]
    result = reranker.rerank("query", chunks, top_k=1)
    
    assert result == [{"content": "A"}]
