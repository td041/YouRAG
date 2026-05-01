import pytest
from pydantic import ValidationError
from src.schema.models import IngestRequest, Citation, QueryRequest, QueryResponse

def test_ingest_request_valid():
    """Kiểm tra IngestRequest với dữ liệu hợp lệ."""
    req = IngestRequest(video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert str(req.video_url) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert req.chunk_size == 500  # Default value
    assert req.overlap == 50      # Default value

def test_ingest_request_invalid_url():
    """Kiểm tra IngestRequest bắt lỗi URL không hợp lệ."""
    with pytest.raises(ValidationError):
        IngestRequest(video_url="not-a-valid-url")

def test_citation_model():
    """Kiểm tra mô hình Citation."""
    citation = Citation(
        video_url="https://youtube.com/test",
        timestamp_start="01:23",
        text_content="This is a test citation",
        relevance_score=0.95
    )
    assert citation.timestamp_start == "01:23"
    assert citation.relevance_score == 0.95

def test_query_request_defaults():
    """Kiểm tra QueryRequest với các giá trị mặc định."""
    req = QueryRequest(query="What is AI?")
    assert req.query == "What is AI?"
    assert req.force_web_search is False
    assert req.filters is None

def test_query_response_structure():
    """Kiểm tra cấu trúc QueryResponse."""
    citations = [
        Citation(video_url="http://x.com", timestamp_start="00:00", text_content="abc", relevance_score=0.9)
    ]
    resp = QueryResponse(
        answer="AI stands for Artificial Intelligence.",
        citations=citations,
        latency_ms=120.5,
        cache_hit=False
    )
    assert resp.answer == "AI stands for Artificial Intelligence."
    assert len(resp.citations) == 1
    assert resp.cache_hit is False
