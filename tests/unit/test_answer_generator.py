"""Tests for AnswerGenerator — context building, mode detection, self-correction."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_llm_client(mocker):
    """Block LLMClient from being instantiated (would load Groq SDK and settings)."""
    mock_cls = mocker.patch("src.engine.generation.answer_generator.LLMClient")
    mock_instance = MagicMock()
    mock_instance.chat_complete.return_value = "Đây là câu trả lời."
    mock_cls.return_value = mock_instance
    return mock_instance


from src.engine.generation.answer_generator import AnswerGenerator  # noqa: E402


def _make_chunk(
    content: str = "Nội dung video.",
    start_time: float = 0.0,
    end_time: float = 60.0,
    context_snippet: str = "",
) -> dict:
    """Helper to create a mock retrieved chunk."""
    return {
        "content": content,
        "metadata": {
            "start_time": start_time,
            "end_time": end_time,
            "context_snippet": context_snippet,
        },
    }


# ── _build_context_string tests ───────────────────────────────────────────

def test_build_context_string_empty_returns_no_info_message():
    """Kiểm tra context string rỗng khi không có chunks."""
    generator = AnswerGenerator()
    result = generator._build_context_string([])
    assert "Không có thông tin" in result


def test_build_context_string_formats_timestamps():
    """Kiểm tra timestamps được format đúng (mm:ss) trong context string."""
    generator = AnswerGenerator()
    chunk = _make_chunk(content="Nội dung test.", start_time=90.0, end_time=150.0)
    result = generator._build_context_string([chunk])
    # 90s = 1:30, 150s = 2:30
    assert "1:30" in result
    assert "2:30" in result


def test_build_context_string_strips_contextual_prefix():
    """Kiểm tra prefix context (dạng 'prefix\\n\\nactual content') bị loại bỏ đúng."""
    generator = AnswerGenerator()
    chunk = _make_chunk(content="Đây là ngữ cảnh do LLM tạo ra.\n\nNội dung thực sự của chunk.")
    result = generator._build_context_string([chunk])
    assert "Nội dung thực sự của chunk." in result
    # Prefix bị loại bỏ
    assert "Đây là ngữ cảnh do LLM tạo ra." not in result


# ── Mode detection tests ───────────────────────────────────────────────────

def test_generate_detects_mindmap_mode(mock_llm_client, mocker):
    """Kiểm tra query chứa 'sơ đồ' kích hoạt mode='mindmap'."""
    mock_pb = mocker.patch("src.engine.generation.answer_generator.PromptBuilder")
    mock_pb.build_system_prompt.return_value = "system prompt"
    mock_pb.build_user_prompt.return_value = "user prompt"

    generator = AnswerGenerator()
    generator.generate(query="Vẽ sơ đồ về RAG", retrieved_chunks=[])

    mock_pb.build_system_prompt.assert_called_once_with(mode="mindmap")


def test_generate_detects_standard_mode(mock_llm_client, mocker):
    """Kiểm tra query thông thường kích hoạt mode='standard'."""
    mock_pb = mocker.patch("src.engine.generation.answer_generator.PromptBuilder")
    mock_pb.build_system_prompt.return_value = "system prompt"
    mock_pb.build_user_prompt.return_value = "user prompt"

    generator = AnswerGenerator()
    generator.generate(query="RAG hoạt động như thế nào?", retrieved_chunks=[])

    mock_pb.build_system_prompt.assert_called_once_with(mode="standard")


# ── Self-correction tests ──────────────────────────────────────────────────

def test_generate_calls_self_correction_when_graph_facts(mock_llm_client):
    """Kiểm tra self-correction được kích hoạt khi có graph_facts."""
    generator = AnswerGenerator()
    generator.generate(
        query="Bao nhiêu rounds?",
        retrieved_chunks=[],
        graph_facts=["The game has 3 rounds"]
    )
    # chat_complete gọi 2 lần: 1 draft + 1 self-correction
    assert mock_llm_client.chat_complete.call_count == 2


def test_generate_skips_self_correction_when_no_graph_facts(mock_llm_client):
    """Kiểm tra self-correction KHÔNG kích hoạt khi graph_facts=None."""
    generator = AnswerGenerator()
    generator.generate(
        query="Câu hỏi bình thường",
        retrieved_chunks=[],
        graph_facts=None
    )
    # chat_complete gọi 1 lần: chỉ draft
    assert mock_llm_client.chat_complete.call_count == 1


# ── Error handling test ────────────────────────────────────────────────────

def test_generate_returns_error_string_on_exception(mock_llm_client):
    """Kiểm tra generate trả về chuỗi bắt đầu bằng 'Lỗi:' khi chat_complete raise."""
    mock_llm_client.chat_complete.side_effect = Exception("API timeout")

    generator = AnswerGenerator()
    result = generator.generate(
        query="test query",
        retrieved_chunks=[]
    )

    assert result.startswith("Lỗi:")
    assert "API timeout" in result
