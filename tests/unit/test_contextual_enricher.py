"""Unit tests for ContextualEnricher (Anthropic Contextual Retrieval)."""

import pytest
import os
from unittest.mock import MagicMock


@pytest.fixture
def cache_dir(tmp_path):
    """Tạo thư mục tạm cho cache."""
    d = tmp_path / "context_cache"
    d.mkdir()
    return str(d)


@pytest.fixture
def mock_llm(mocker):
    """Mock LLMClient and Redis (no real I/O in unit tests)."""
    mock_cls = mocker.patch("src.engine.ingestion.contextual_enricher.LLMClient")
    mock_instance = MagicMock()
    mock_instance.chat_complete.return_value = "This is context."
    mock_cls.return_value = mock_instance

    # Disable Redis so tests use local file fallback
    mocker.patch("src.engine.ingestion.contextual_enricher.get_redis", return_value=None)

    return mock_instance


from src.engine.ingestion.contextual_enricher import ContextualEnricher  # noqa: E402


def test_prepare_full_text_no_truncation():
    """Kiểm tra văn bản ngắn không bị truncate."""
    text = "Short text"
    result = ContextualEnricher._prepare_full_text(text)
    assert result == text


def test_prepare_full_text_truncation():
    """Kiểm tra văn bản quá dài bị truncate ở giữa."""
    long_text = "A" * 15000
    result = ContextualEnricher._prepare_full_text(long_text)
    assert len(result) < 15000
    assert "[... transcript truncated" in result


def test_enrich_single_cache_hit(mock_llm, cache_dir, mocker):
    """Kiểm tra cache hit không gọi LLM."""
    enricher = ContextualEnricher(cache_dir=cache_dir, llm_client=mock_llm)
    
    # Giả lập cache tồn tại
    mocker.patch.object(enricher, "_read_cache", return_value="Cached context")
    
    chunk = {"content": "Original content", "metadata": {"video_id": "vid1"}}
    result = enricher._enrich_single("Full text", chunk, 0, "vid1")
    
    assert "Cached context" in result["content"]
    assert mock_llm.chat_complete.call_count == 0


def test_enrich_single_cache_miss_calls_llm(mock_llm, cache_dir):
    """Kiểm tra cache miss gọi LLM và lưu vào cache."""
    enricher = ContextualEnricher(cache_dir=cache_dir, llm_client=mock_llm)
    
    chunk = {"content": "Original content", "metadata": {"video_id": "vid1"}}
    result = enricher._enrich_single("Full text", chunk, 0, "vid1")
    
    assert "This is context." in result["content"]
    assert mock_llm.chat_complete.call_count == 1
    
    # Kiểm tra xem file cache có được tạo không
    assert len(os.listdir(cache_dir)) > 0


def test_enrich_multiple_chunks(mock_llm, cache_dir):
    """Kiểm tra enrich nhiều chunks song song."""
    enricher = ContextualEnricher(cache_dir=cache_dir, llm_client=mock_llm)
    
    chunks = [
        {"content": "C1", "metadata": {"video_id": "v1"}},
        {"content": "C2", "metadata": {"video_id": "v1"}}
    ]
    
    results = enricher.enrich("Full text", chunks)
    
    assert len(results) == 2
    assert "This is context." in results[0]["content"]
    assert "This is context." in results[1]["content"]
    assert mock_llm.chat_complete.call_count == 2


def test_enrich_graceful_degrade_on_llm_error(mock_llm, cache_dir):
    """Kiểm tra nếu LLM lỗi thì vẫn trả về chunk gốc."""
    mock_llm.chat_complete.side_effect = Exception("Groq down")
    enricher = ContextualEnricher(cache_dir=cache_dir, llm_client=mock_llm)
    
    chunk = {"content": "Original", "metadata": {"video_id": "v1"}}
    result = enricher._enrich_single("Full", chunk, 0, "v1")
    
    assert result["content"] == "Original"
    assert result["metadata"]["has_context"] is False
