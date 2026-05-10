"""Tests for PromptBuilder — pure static class, no mocking needed."""

import pytest
from src.engine.generation.prompt_builder import PromptBuilder


def test_system_prompt_standard_mode():
    """Kiểm tra system prompt mode 'standard' có hướng dẫn tiếng Việt."""
    result = PromptBuilder.build_system_prompt(mode="standard")
    assert "tiếng Việt" in result
    # Standard mode không có Mermaid
    assert "Mermaid" not in result
    assert "graph TD" not in result


def test_system_prompt_mindmap_mode():
    """Kiểm tra system prompt mode 'mindmap' có syntax Mermaid."""
    result = PromptBuilder.build_system_prompt(mode="mindmap")
    assert "Mermaid" in result
    assert "graph TD" in result


def test_system_prompt_default_is_standard():
    """Kiểm tra default (không truyền mode) trả về cùng kết quả standard mode."""
    default_result = PromptBuilder.build_system_prompt()
    standard_result = PromptBuilder.build_system_prompt(mode="standard")
    assert default_result == standard_result


def test_user_prompt_contains_query():
    """Kiểm tra user prompt chứa câu hỏi của người dùng."""
    result = PromptBuilder.build_user_prompt(
        query="Điều gì được nói về RAG?",
        context_str="Một số nội dung về RAG."
    )
    assert "Điều gì được nói về RAG?" in result


def test_user_prompt_no_optional_sections_when_none():
    """Kiểm tra các section tùy chọn không xuất hiện khi truyền None."""
    result = PromptBuilder.build_user_prompt(
        query="test question",
        context_str="some context",
        global_summary=None,
        graph_facts=None,
        graph_summary=None
    )
    assert "TỔNG QUAN" not in result
    assert "BẢN ĐỒ THỰC THỂ" not in result
    assert "DỮ LIỆU ĐỒ THỊ" not in result


def test_user_prompt_includes_summary_when_provided():
    """Kiểm tra global_summary xuất hiện trong prompt khi được cung cấp."""
    result = PromptBuilder.build_user_prompt(
        query="test",
        context_str="context",
        global_summary="Video này nói về học máy và AI."
    )
    assert "Video này nói về học máy và AI." in result
    assert "TỔNG QUAN" in result


def test_user_prompt_includes_graph_facts():
    """Kiểm tra graph_facts xuất hiện trong prompt khi được cung cấp."""
    facts = ["Entity A → relates to → Entity B", "RAG uses 3 rounds of retrieval"]
    result = PromptBuilder.build_user_prompt(
        query="test",
        context_str="context",
        graph_facts=facts
    )
    assert "Entity A → relates to → Entity B" in result
    assert "RAG uses 3 rounds of retrieval" in result
    assert "DỮ LIỆU ĐỒ THỊ" in result


def test_self_correction_prompt_contains_draft_and_facts():
    """Kiểm tra self-correction prompt chứa câu trả lời nháp và graph facts."""
    draft = "AI được sử dụng trong 5 rounds."
    facts = ["The game has 3 rounds", "Players use tokens"]
    result = PromptBuilder.build_self_correction_prompt(
        query="Bao nhiêu rounds?",
        draft_answer=draft,
        graph_facts=facts
    )
    assert draft in result
    assert "The game has 3 rounds" in result
    assert "Players use tokens" in result
    # Phải có hướng dẫn sửa lỗi
    assert "mâu thuẫn" in result or "sửa" in result
