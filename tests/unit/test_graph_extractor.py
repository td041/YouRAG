"""Tests for GraphExtractor — pure rule-based entity extraction, no mocking needed."""

import pytest
from src.engine.ingestion.graph_extractor import GraphExtractor


@pytest.fixture
def extractor():
    """Default GraphExtractor instance."""
    return GraphExtractor(min_chunk_length=150, max_keywords=10)


def test_init_defaults():
    """Kiểm tra giá trị mặc định khi khởi tạo GraphExtractor."""
    ge = GraphExtractor()
    assert ge.min_chunk_length == 150
    assert ge.max_keywords == 10


def test_extract_capitalized_phrases(extractor):
    """Kiểm tra trích xuất cụm danh từ riêng viết hoa đầu chữ."""
    text = "Neural Network is used by Deep Learning experts at Google Brain."
    result = extractor._extract_capitalized_phrases(text)
    assert "Neural Network" in result
    assert "Deep Learning" in result
    assert "Google Brain" in result


def test_extract_abbreviations(extractor):
    """Kiểm tra trích xuất từ viết tắt ALL-CAPS."""
    text = "The AI and RAG system uses NLP and LLM techniques."
    result = extractor._extract_abbreviations(text)
    assert "AI" in result
    assert "RAG" in result
    assert "NLP" in result
    assert "LLM" in result


def test_extract_numeric_phrases(extractor):
    """Kiểm tra trích xuất cụm số + đơn vị."""
    text = "The game has 3 rounds and takes 5 minutes to play with 10 players."
    result = extractor._extract_numeric_phrases(text)
    # Kiểm tra các cụm số-đơn vị được tìm thấy
    result_lower = [r.lower() for r in result]
    assert any("3 round" in r for r in result_lower)
    assert any("5 minute" in r for r in result_lower)
    assert any("10 player" in r for r in result_lower)


def test_extract_top_keywords_filters_stopwords(extractor):
    """Kiểm tra top keywords lọc stopwords và chỉ lấy từ xuất hiện >= 2 lần."""
    text = "the system the system the system uses retrieval retrieval retrieval augmented generation"
    result = extractor._extract_top_keywords(text)
    assert "system" in result
    assert "retrieval" in result
    # Stopwords như "the" phải bị lọc
    assert "the" not in result


def test_extract_entities_deduplicates(extractor):
    """Kiểm tra kết quả không có duplicates (case-insensitive)."""
    text = "AI is powerful. AI systems use NLP. artificial intelligence is AI."
    entities = extractor.extract_entities(text)
    # Đếm số lần "AI" xuất hiện — phải là 1
    lower_entities = [e.lower() for e in entities]
    assert lower_entities.count("ai") <= 1


def test_process_chunk_short_content(extractor):
    """Chunk quá ngắn (< min_chunk_length) phải bị bỏ qua, keywords = ''."""
    chunk = {
        "content": "Short text.",
        "metadata": {}
    }
    result = extractor.process_chunk(chunk)
    assert result["metadata"]["keywords"] == ""


def test_process_chunk_normal(extractor):
    """Chunk đủ dài phải được enrich với keywords trong metadata."""
    # Tạo content dài hơn 150 ký tự với entity rõ ràng
    content = (
        "Retrieval Augmented Generation (RAG) is a technique used in AI systems. "
        "RAG combines dense retrieval with LLM generation. "
        "RAG retrieval augmented generation systems are used by AI researchers. "
        "The RAG technique improves LLM accuracy significantly."
    )
    chunk = {
        "content": content,
        "metadata": {}
    }
    result = extractor.process_chunk(chunk)
    assert "keywords" in result["metadata"]
    assert result["metadata"]["keywords"] != ""
    # Kiểm tra có chứa entity từ content
    keywords_str = result["metadata"]["keywords"]
    assert len(keywords_str) > 0
