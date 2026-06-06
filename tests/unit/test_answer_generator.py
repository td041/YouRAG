"""Tests for AnswerGenerator — context building, mode detection, self-correction, citation grounding."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_llm_client(mocker):
    """Block LLMClient from being instantiated (would load Groq SDK and settings)."""
    mock_cls = mocker.patch("src.engine.generation.answer_generator.LLMClient")
    mock_instance = MagicMock()
    mock_instance.chat_complete.return_value = "Đây là câu trả lời."
    mock_instance.chat_complete_stream.return_value = iter(["Đây ", "là ", "câu trả lời."])
    mock_cls.return_value = mock_instance
    return mock_instance


@pytest.fixture(autouse=True)
def mock_semantic_cache(mocker):
    """SemanticCache removed from AnswerGenerator — caching now handled at API layer."""
    return MagicMock()


from src.engine.generation.answer_generator import AnswerGenerator, _NO_CONTEXT_REPLY  # noqa: E402


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
    """Kiểm tra prefix context (dạng 'prefix\n\nactual content') bị loại bỏ đúng."""
    generator = AnswerGenerator()
    chunk = _make_chunk(content="Đây là ngữ cảnh do LLM tạo ra.\n\nNội dung thực sự của chunk.")
    result = generator._build_context_string([chunk])
    assert "Nội dung thực sự của chunk." in result
    assert "Đây là ngữ cảnh do LLM tạo ra." not in result


# ── No-context early return ────────────────────────────────────────────────

def test_generate_returns_no_context_reply_when_chunks_empty(mock_llm_client, mock_semantic_cache):
    """Kiểm tra generate trả về no-context reply khi không có chunks, không gọi LLM."""
    generator = AnswerGenerator()
    result = generator.generate(query="test", retrieved_chunks=[])
    assert result == _NO_CONTEXT_REPLY
    mock_llm_client.chat_complete.assert_not_called()


def test_generate_stream_yields_no_context_reply_when_chunks_empty(mock_llm_client, mock_semantic_cache):
    """Kiểm tra generate_stream yield no-context reply khi chunks rỗng."""
    generator = AnswerGenerator()
    results = list(generator.generate_stream(query="test", retrieved_chunks=[]))
    assert results == [_NO_CONTEXT_REPLY]
    mock_llm_client.chat_complete_stream.assert_not_called()


# ── Citation Grounding tests ───────────────────────────────────────────────

def test_validate_citations_removes_ungrounded_timestamps():
    """Kiểm tra [mm:ss] không có trong chunks bị xóa khỏi câu trả lời."""
    generator = AnswerGenerator()
    chunks = [_make_chunk(start_time=0.0, end_time=60.0)]
    answer = "Đây là điểm quan trọng [0:30] và điểm bịa [5:00]."
    result = generator._validate_citations(answer, chunks)
    assert "[0:30]" in result
    assert "[5:00]" not in result


def test_validate_citations_keeps_grounded_timestamps():
    """Kiểm tra [mm:ss] hợp lệ được giữ nguyên."""
    generator = AnswerGenerator()
    chunks = [_make_chunk(start_time=60.0, end_time=120.0)]
    answer = "Thông tin tại [1:30] rất quan trọng."
    result = generator._validate_citations(answer, chunks)
    assert "[1:30]" in result


def test_validate_citations_empty_chunks_returns_unchanged():
    """Kiểm tra khi chunks rỗng, answer không bị thay đổi."""
    generator = AnswerGenerator()
    answer = "Câu trả lời với [5:00] timestamp."
    result = generator._validate_citations(answer, [])
    assert result == answer


# ── Mode detection tests ───────────────────────────────────────────────────

def test_generate_detects_mindmap_mode(mock_llm_client, mock_semantic_cache, mocker):
    """Kiểm tra query chứa 'sơ đồ' kích hoạt mode='mindmap'."""
    mock_pb = mocker.patch("src.engine.generation.answer_generator.PromptBuilder")
    mock_pb.build_system_prompt.return_value = "system prompt"
    mock_pb.build_user_prompt.return_value = "user prompt"

    generator = AnswerGenerator()
    generator.generate(query="Vẽ sơ đồ về RAG", retrieved_chunks=[_make_chunk()])

    mock_pb.build_system_prompt.assert_called_once_with(mode="mindmap")


def test_generate_detects_standard_mode(mock_llm_client, mock_semantic_cache, mocker):
    """Kiểm tra query thông thường kích hoạt mode='standard'."""
    mock_pb = mocker.patch("src.engine.generation.answer_generator.PromptBuilder")
    mock_pb.build_system_prompt.return_value = "system prompt"
    mock_pb.build_user_prompt.return_value = "user prompt"

    generator = AnswerGenerator()
    generator.generate(query="RAG hoạt động như thế nào?", retrieved_chunks=[_make_chunk()])

    mock_pb.build_system_prompt.assert_called_once_with(mode="standard")


# ── Self-correction tests ──────────────────────────────────────────────────

def test_generate_calls_self_correction_when_graph_facts(mock_llm_client, mock_semantic_cache):
    """Kiểm tra self-correction được kích hoạt khi có graph_facts."""
    generator = AnswerGenerator()
    generator.generate(
        query="Bao nhiêu rounds?",
        retrieved_chunks=[_make_chunk()],
        graph_facts=["The game has 3 rounds"]
    )
    # chat_complete gọi 2 lần: 1 draft + 1 self-correction
    assert mock_llm_client.chat_complete.call_count == 2


def test_generate_skips_self_correction_when_no_graph_facts(mock_llm_client, mock_semantic_cache):
    """Kiểm tra self-correction KHÔNG kích hoạt khi graph_facts=None."""
    generator = AnswerGenerator()
    generator.generate(
        query="Câu hỏi bình thường",
        retrieved_chunks=[_make_chunk()],
        graph_facts=None
    )
    # chat_complete gọi 1 lần: chỉ draft
    assert mock_llm_client.chat_complete.call_count == 1


# ── Error handling test ────────────────────────────────────────────────────

def test_generate_returns_error_string_on_exception(mock_llm_client, mock_semantic_cache):
    """Kiểm tra generate trả về chuỗi bắt đầu bằng 'Lỗi:' khi chat_complete raise."""
    mock_llm_client.chat_complete.side_effect = Exception("API timeout")

    generator = AnswerGenerator()
    result = generator.generate(
        query="test query",
        retrieved_chunks=[_make_chunk()]
    )

    assert result.startswith("Lỗi:")
    assert "API timeout" in result


# ── Caching tests ─────────────────────────────────────────────────────────
# Note: caching is now handled at the API layer (main.py), not inside AnswerGenerator.

def test_generate_calls_llm_when_chunks_present(mock_llm_client, mock_semantic_cache):
    """Kiểm tra generate gọi LLM khi có chunks (cache handled by API layer now)."""
    mock_llm_client.chat_complete.return_value = "LLM answer"

    generator = AnswerGenerator()
    result = generator.generate(query="hello", retrieved_chunks=[_make_chunk()])

    assert result == "LLM answer"
    assert mock_llm_client.chat_complete.call_count == 1


# ── Streaming tests ────────────────────────────────────────────────────────

def test_generate_stream_yields_chunks(mock_llm_client, mock_semantic_cache):
    """Kiểm tra generate_stream yield từng chunk từ LLM (no graph_facts → fast path)."""
    mock_llm_client.chat_complete_stream.return_value = iter(["Hello", " World"])

    generator = AnswerGenerator()
    results = list(generator.generate_stream(query="test", retrieved_chunks=[_make_chunk()]))

    assert "Hello" in results
    assert " World" in results


def test_generate_stream_error_yields_error(mock_llm_client, mock_semantic_cache):
    """Kiểm tra generate_stream yield lỗi khi LLM crash."""
    mock_llm_client.chat_complete_stream.side_effect = Exception("Stream crash")

    generator = AnswerGenerator()
    results = list(generator.generate_stream(query="test", retrieved_chunks=[_make_chunk()]))

    assert len(results) == 1
    assert "Error" in results[0] or "Lỗi" in results[0]


def test_generate_stream_with_graph_facts_uses_stream(mock_llm_client, mock_semantic_cache):
    """graph_facts are injected into the prompt — generate_stream always uses true LLM streaming."""
    mock_llm_client.chat_complete_stream.return_value = iter(["Graph-aware answer"])

    generator = AnswerGenerator()
    results = list(generator.generate_stream(
        query="test", retrieved_chunks=[_make_chunk()],
        graph_facts=["Python is great"],
        graph_summary="Summary"
    ))

    # Always streams directly regardless of graph_facts
    mock_llm_client.chat_complete_stream.assert_called_once()
    mock_llm_client.chat_complete.assert_not_called()
    assert "Graph-aware answer" in "".join(results)


def test_generate_with_chat_history(mock_llm_client, mock_semantic_cache, mocker):
    """Kiểm tra generate truyền chat_history vào prompt builder."""
    mock_pb = mocker.patch("src.engine.generation.answer_generator.PromptBuilder")
    mock_pb.build_system_prompt.return_value = "sys"
    mock_pb.build_user_prompt.return_value = "user"

    generator = AnswerGenerator()
    generator.generate(query="test", retrieved_chunks=[_make_chunk()], chat_history="User: hi\nAI: hello")

    mock_pb.build_user_prompt.assert_called_once()
    call_kwargs = mock_pb.build_user_prompt.call_args
    assert call_kwargs.kwargs.get("chat_history") == "User: hi\nAI: hello"
