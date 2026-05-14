"""Unit tests for SemanticChunker (SOTA Pause-aware & Semantic Grouping)."""

import pytest
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_encoder(mocker):
    """Mock SentenceTransformer."""
    mock_cls = mocker.patch("src.engine.ingestion.chunker.SentenceTransformer")
    mock_instance = MagicMock()
    # Mock encode to return dummy vectors
    mock_instance.encode.side_effect = lambda texts: [np.random.rand(16) for _ in texts]
    mock_cls.return_value = mock_instance
    return mock_instance


from src.engine.ingestion.chunker import SemanticChunker, cosine_similarity  # noqa: E402


def test_cosine_similarity():
    """Kiểm tra tính cosine similarity."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)
    
    v3 = np.array([0, 1, 0])
    assert cosine_similarity(v1, v3) == pytest.approx(0.0)


def test_correct_punctuation_calls_llm(mocker):
    """Kiểm tra gọi LLM để phục hồi dấu câu."""
    mock_llm = MagicMock()
    mock_llm.chat_complete.return_value = "Corrected text."
    
    chunker = SemanticChunker(llm_client=mock_llm)
    result = chunker._correct_punctuation("text")
    
    assert result == "Corrected text."
    mock_llm.chat_complete.assert_called_once()


def test_build_atomic_sentences_pause_detection():
    """Kiểm tra ngắt câu dựa trên khoảng lặng (pause)."""
    chunker = SemanticChunker(pause_threshold_sec=1.0)
    
    transcript = [
        {"text": "Hello", "start": 0.0, "duration": 1.0},
        {"text": "World", "start": 3.0, "duration": 1.0} # Pause 2s > 1s
    ]
    
    sentences = chunker._build_atomic_sentences(transcript)
    
    assert len(sentences) == 2
    assert sentences[0]["text"] == "Hello"
    assert sentences[1]["text"] == "World"


def test_build_atomic_sentences_punctuation_detection():
    """Kiểm tra ngắt câu dựa trên dấu câu."""
    chunker = SemanticChunker()
    
    transcript = [
        {"text": "Hello.", "start": 0.0, "duration": 1.0},
        {"text": "How are you?", "start": 1.0, "duration": 1.0}
    ]
    
    sentences = chunker._build_atomic_sentences(transcript)
    
    assert len(sentences) == 2
    assert "Hello" in sentences[0]["text"]


def test_chunk_document_semantic_splitting(mock_encoder):
    """Kiểm tra logic băm chunk dựa trên ngữ nghĩa (similarities)."""
    chunker = SemanticChunker(percentile_threshold=50) # Ngưỡng cao để dễ ngắt
    
    metadata = {"video_id": "v1"}
    transcript = [
        {"text": "Topic A sentence 1.", "start": 0.0, "duration": 1.0},
        {"text": "Topic A sentence 2.", "start": 1.0, "duration": 1.0},
        {"text": "Topic B sentence 1.", "start": 2.0, "duration": 1.0},
        {"text": "Topic B sentence 2.", "start": 3.0, "duration": 1.0},
    ]
    
    # Mock similarities: A1-A2 (high), A2-B1 (low), B1-B2 (high)
    def mock_encode(texts):
        if "Topic A" in texts[0]:
            vA = np.array([1, 0])
            vB = np.array([0, 1])
            return [vA, vA, vB, vB]
        return [np.random.rand(2) for _ in texts]
        
    mock_encoder.encode.side_effect = mock_encode
    
    # Ép min_chars nhỏ để ngắt được
    chunker.min_chars = 5
    
    chunks = chunker.chunk_document(metadata, transcript)
    
    # Kỳ vọng 2 chunks: Topic A và Topic B
    assert len(chunks) >= 2
    assert "Topic A" in chunks[0]["content"]


def test_chunk_document_length_limit(mock_encoder):
    """Kiểm tra ngắt chunk khi quá dài (max_chars)."""
    chunker = SemanticChunker(max_chars_per_chunk=20)
    
    metadata = {"video_id": "v1"}
    transcript = [
        {"text": "This is a very long sentence that exceeds limit.", "start": 0.0, "duration": 1.0},
        {"text": "Next sentence.", "start": 1.0, "duration": 1.0},
    ]
    
    chunks = chunker.chunk_document(metadata, transcript)
    
    assert len(chunks) == 2


def test_chunk_document_single_sentence():
    """Kiểm tra chunk document với transcript quá ngắn (1 câu)."""
    chunker = SemanticChunker()
    metadata = {"video_id": "123", "title": "Test"}
    raw = [{"text": "Just one sentence.", "start": 0.0, "duration": 1.0}]
    
    chunks = chunker.chunk_document(metadata, raw)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Just one sentence."
