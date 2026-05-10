import pytest
import numpy as np
from unittest.mock import MagicMock

# Vá lỗi SentenceTransformer trước khi import SemanticChunker để không bị lỗi load model
@pytest.fixture(autouse=True)
def mock_sentence_transformer(mocker):
    mock_st = mocker.patch("src.engine.ingestion.chunker.SentenceTransformer")
    # Giả lập encoder trả về vector ngẫu nhiên
    mock_instance = MagicMock()
    mock_instance.encode.side_effect = lambda texts: [np.random.rand(10) for _ in texts]
    mock_st.return_value = mock_instance
    return mock_st

from src.engine.ingestion.chunker import cosine_similarity, SemanticChunker  # noqa: E402

def test_cosine_similarity():
    """Kiểm tra hàm tính cosine similarity toán học."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    vec3 = np.array([0.0, 1.0, 0.0])
    
    assert np.isclose(cosine_similarity(vec1, vec2), 1.0)
    assert np.isclose(cosine_similarity(vec1, vec3), 0.0)
    
    # Test zero vectors
    assert cosine_similarity(np.zeros(3), vec1) == 0.0

def test_semantic_chunker_init():
    """Kiểm tra khởi tạo SemanticChunker."""
    chunker = SemanticChunker(percentile_threshold=20, min_chars_per_chunk=100)
    assert chunker.percentile_threshold == 20
    assert chunker.min_chars == 100
    assert chunker.encoder is not None  # Đã được mock

def test_build_atomic_sentences():
    """Kiểm tra việc gộp dòng transcript dựa trên dấu câu và khoảng lặng."""
    chunker = SemanticChunker(pause_threshold_sec=1.5)
    
    raw_transcript = [
        {"text": "Hello world", "start": 0.0, "duration": 1.0},
        {"text": "This is a test.", "start": 1.5, "duration": 2.0},  # Có dấu chấm
        {"text": "Another sentence", "start": 3.6, "duration": 1.0}, # Khoảng lặng < 1.5s
        {"text": "with no punctuation", "start": 4.7, "duration": 1.0},
        {"text": "New topic entirely", "start": 10.0, "duration": 2.0} # Khoảng lặng > 1.5s (10.0 - 5.7 = 4.3s)
    ]
    
    atomic = chunker._build_atomic_sentences(raw_transcript)
    
    # Kỳ vọng:
    # 1. "Hello world This is a test." (Gộp do ko có pause dài, ngắt do có dấu chấm)
    # 2. "Another sentence with no punctuation" (Gộp do ko có pause, ngắt do câu sau có pause dài)
    # 3. "New topic entirely" (Câu cuối)
    
    assert len(atomic) == 3
    assert atomic[0]["text"] == "Hello world This is a test."
    assert atomic[0]["start"] == 0.0
    assert atomic[0]["end"] == 3.5
    
    assert atomic[1]["text"] == "Another sentence with no punctuation"
    assert atomic[1]["start"] == 3.6
    assert atomic[1]["end"] == 5.7
    
    assert atomic[2]["text"] == "New topic entirely"
    assert atomic[2]["start"] == 10.0

def test_chunk_document_empty():
    """Kiểm tra chunk document với transcript rỗng."""
    chunker = SemanticChunker()
    metadata = {"video_id": "123"}
    chunks = chunker.chunk_document(metadata, [])
    assert chunks == []

def test_chunk_document_single_sentence():
    """Kiểm tra chunk document với transcript quá ngắn (1 câu)."""
    chunker = SemanticChunker()
    metadata = {"video_id": "123", "title": "Test"}
    raw = [{"text": "Just one sentence.", "start": 0.0, "duration": 1.0}]
    
    chunks = chunker.chunk_document(metadata, raw)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Just one sentence."
    assert chunks[0]["metadata"]["video_id"] == "123"
    assert chunks[0]["metadata"]["chunk_index"] == 0
