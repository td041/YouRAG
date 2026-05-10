"""Tests for LLMClient — provider selection, retry logic, rate-limit backoff."""

import pytest
import time
from unittest.mock import MagicMock, patch, call

# Mock settings và groq TRƯỚC KHI import LLMClient để tránh module-level side effects
@pytest.fixture(autouse=True)
def mock_settings_and_providers(mocker):
    """Mock settings và tất cả provider SDKs."""
    mock_s = MagicMock()
    mock_s.LLM_PROVIDER = "groq"
    mock_s.GROQ_API_KEY.get_secret_value.return_value = "fake_groq_key"
    mock_s.OPENAI_API_KEY = None
    mock_s.LLM_CONTEXTUAL_MODEL = "llama-3.1-8b-instant"
    mock_s.LLM_MODEL_NAME = "llama-3.3-70b-versatile"
    mock_s.OLLAMA_BASE_URL = "http://localhost:11434/v1"
    mocker.patch("src.engine.generation.llm_client.settings", mock_s)
    return mock_s


from src.engine.generation.llm_client import LLMClient  # noqa: E402


def _make_response(content: str) -> MagicMock:
    """Helper to create a mock LLM API response."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp


# ── Provider selection tests ───────────────────────────────────────────────

def test_provider_selection_groq(mock_settings_and_providers):
    """Kiểm tra LLMClient chọn Groq khi LLM_PROVIDER='groq'."""
    mock_settings_and_providers.LLM_PROVIDER = "groq"
    with patch("groq.Groq"):
        client = LLMClient(provider="groq")
    assert client.provider == "groq"


def test_provider_selection_ollama(mock_settings_and_providers):
    """Kiểm tra LLMClient chọn Ollama và dùng đúng model.

    LLMClient.__init__ uses `from openai import OpenAI` inline when provider=ollama.
    We mock the openai module in sys.modules so the inline import resolves to our mock.
    """
    import sys
    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI = MagicMock(return_value=MagicMock())
    mock_settings_and_providers.LLM_PROVIDER = "ollama"
    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        client = LLMClient(provider="ollama")
    assert client.provider == "ollama"
    assert client.model == "qwen2.5:7b-instruct-q4_K_M"


def test_invalid_provider_raises():
    """Kiểm tra ValueError khi provider không được hỗ trợ."""
    with pytest.raises(ValueError, match="Provider không hỗ trợ"):
        LLMClient(provider="unknown_provider")


# ── chat_complete success tests ────────────────────────────────────────────

def test_chat_complete_success():
    """Kiểm tra chat_complete trả về content khi API thành công."""
    with patch("groq.Groq") as mock_groq_cls:
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        mock_groq_instance.chat.completions.create.return_value = _make_response("Hello world")

        client = LLMClient(provider="groq")
        result = client.chat_complete("Tell me about RAG")

    assert result == "Hello world"


# ── Retry logic tests ──────────────────────────────────────────────────────

def test_chat_complete_retries_on_exception():
    """Kiểm tra chat_complete retry khi gặp exception, thành công ở lần thứ 3."""
    with patch("groq.Groq") as mock_groq_cls, \
         patch("time.sleep") as mock_sleep:
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        mock_groq_instance.chat.completions.create.side_effect = [
            Exception("network error"),
            Exception("network error"),
            _make_response("Success after retries"),
        ]

        client = LLMClient(provider="groq")
        result = client.chat_complete("test prompt")

    assert result == "Success after retries"
    assert mock_groq_instance.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2  # sleep after each failure except last


def test_chat_complete_rate_limit_parses_wait_time():
    """Kiểm tra parse thời gian chờ từ message Groq rate limit ('try again in X.Xs')."""
    with patch("groq.Groq") as mock_groq_cls, \
         patch("time.sleep") as mock_sleep:
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        mock_groq_instance.chat.completions.create.side_effect = [
            Exception("Please try again in 2.5s. Rate limit exceeded."),
            _make_response("Rate limit recovered"),
        ]

        client = LLMClient(provider="groq")
        result = client.chat_complete("test prompt")

    assert result == "Rate limit recovered"
    # Should sleep for 2.5 + 0.5 = 3.0 seconds
    mock_sleep.assert_called_once_with(3.0)


def test_chat_complete_raises_after_max_retries():
    """Kiểm tra RuntimeError được raise sau khi hết số lần retry."""
    with patch("groq.Groq") as mock_groq_cls, \
         patch("time.sleep"):
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        mock_groq_instance.chat.completions.create.side_effect = Exception("persistent failure")

        client = LLMClient(provider="groq", max_retries=3)
        with pytest.raises(RuntimeError, match="LLM call thất bại"):
            client.chat_complete("test prompt")

    assert mock_groq_instance.chat.completions.create.call_count == 3
