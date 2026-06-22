"""Tests for QueryExpander — query paraphrasing với LLM fallback."""

from unittest.mock import MagicMock

# QueryExpander uses lazy LLM init, so no global mock needed
from src.engine.retrieval.query_expander import QueryExpander  # noqa: E402


def _make_expander(llm_response: str) -> QueryExpander:
    expander = QueryExpander(n=2)
    mock_llm = MagicMock()
    mock_llm.chat_complete.return_value = llm_response
    expander._llm = mock_llm  # bypass lazy init
    return expander


def test_expand_returns_list_on_valid_json():
    expander = _make_expander('["câu hỏi A", "câu hỏi B"]')
    result = expander.expand("video nói về gì?")
    assert result == ["câu hỏi A", "câu hỏi B"]


def test_expand_strips_markdown_code_block():
    expander = _make_expander('```json\n["q1", "q2"]\n```')
    result = expander.expand("test query")
    assert result == ["q1", "q2"]


def test_expand_caps_at_n():
    expander = _make_expander('["q1", "q2", "q3", "q4"]')
    result = expander.expand("test")
    assert len(result) <= 2


def test_expand_returns_empty_on_llm_error():
    expander = QueryExpander(n=2)
    mock_llm = MagicMock()
    mock_llm.chat_complete.side_effect = Exception("Rate limit exceeded")
    expander._llm = mock_llm
    result = expander.expand("test query")
    assert result == []


def test_expand_returns_empty_on_invalid_json():
    expander = _make_expander("not valid json at all")
    result = expander.expand("test query")
    assert result == []


def test_expand_returns_empty_on_non_list_json():
    expander = _make_expander('{"q1": "value"}')
    result = expander.expand("test query")
    assert result == []


def test_expand_filters_non_string_items():
    expander = _make_expander('["valid string here", 123, null, "also valid one"]')
    result = expander.expand("test")
    assert all(isinstance(q, str) for q in result)
    assert "valid string here" in result
