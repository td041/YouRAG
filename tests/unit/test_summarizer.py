"""Unit tests for VideoSummarizer."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_deps(mocker):
    """Mock LLMClient and db_instance."""
    mock_llm_cls = mocker.patch("src.engine.generation.summarizer.LLMClient")
    mock_llm = MagicMock()
    mock_llm.chat_complete.return_value = "Bản tóm tắt video."
    mock_llm_cls.return_value = mock_llm
    
    mock_db = MagicMock()
    mocker.patch("src.engine.generation.summarizer.db_instance", mock_db)
    
    return mock_llm, mock_db


from src.engine.generation.summarizer import VideoSummarizer  # noqa: E402


def test_summarize_empty_collection(mock_deps):
    """Kiểm tra summarize trả về None nếu collection không có records."""
    mock_llm, mock_db = mock_deps
    mock_db.client.scroll.return_value = ([], None)
    
    summarizer = VideoSummarizer()
    result = summarizer.summarize("empty_vid")
    
    assert result is None


def test_summarize_calls_llm_with_formatted_transcript(mock_deps):
    """Kiểm tra summarize format kịch bản và gọi LLM."""
    mock_llm, mock_db = mock_deps
    
    # Mock 2 chunks
    fake_rec1 = MagicMock()
    fake_rec1.payload = {"text": "Hello world", "chunk_index": 0, "start_time": 0.0}
    fake_rec2 = MagicMock()
    fake_rec2.payload = {"text": "Goodbye", "chunk_index": 1, "start_time": 60.0}
    
    mock_db.client.scroll.return_value = ([fake_rec1, fake_rec2], None)
    
    summarizer = VideoSummarizer()
    result = summarizer.summarize("vid_123")
    
    assert result == "Bản tóm tắt video."
    mock_llm.chat_complete.assert_called_once()
    prompt = mock_llm.chat_complete.call_args.kwargs["prompt"]
    assert "[0:00] Hello world" in prompt
    assert "[1:00] Goodbye" in prompt


def test_summarize_sampling_logic(mock_deps):
    """Kiểm tra logic trích mẫu (Uniform Sampling) cho video quá dài (>12 chunks)."""
    mock_llm, mock_db = mock_deps
    
    # Tạo 20 chunks
    fake_recs = []
    for i in range(20):
        m = MagicMock()
        m.payload = {"text": f"Text {i}", "chunk_index": i, "start_time": float(i)}
        fake_recs.append(m)
        
    mock_db.client.scroll.return_value = (fake_recs, None)
    
    summarizer = VideoSummarizer()
    summarizer.summarize("long_vid")
    
    prompt = mock_llm.chat_complete.call_args.kwargs["prompt"]
    # Kiểm tra xem có 12 mốc thời gian không
    assert prompt.count("[") >= 12


def test_summarize_stream_yields_chunks(mock_deps):
    """Kiểm tra summarize_stream trả về generator và yield đúng dữ liệu."""
    mock_llm, mock_db = mock_deps
    
    # Mock streaming response
    mock_llm.chat_complete_stream.return_value = iter(["Bản", " tóm", " tắt"])
    
    fake_rec = MagicMock()
    fake_rec.payload = {"text": "Hello", "chunk_index": 0, "start_time": 0.0}
    mock_db.client.scroll.return_value = ([fake_rec], None)
    
    summarizer = VideoSummarizer()
    generator = summarizer.summarize_stream("vid_123")
    
    results = list(generator)
    assert results == ["Bản", " tóm", " tắt"]
