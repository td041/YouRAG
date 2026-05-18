"""Tests for LateChunkingEmbedder — Jina AI late chunking API."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_settings(mocker):
    """Mock settings với JINA_API_KEY hợp lệ."""
    mock_s = MagicMock()
    mock_s.JINA_API_KEY.get_secret_value.return_value = "fake_jina_key"
    mocker.patch("src.engine.ingestion.late_chunker.settings", mock_s)
    return mock_s


from src.engine.ingestion.late_chunker import LateChunkingEmbedder  # noqa: E402


def _make_jina_response(n: int) -> MagicMock:
    """Tạo mock response Jina với n embeddings 1024-dim."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"embedding": [0.1] * 1024} for _ in range(n)]
    }
    return mock_resp


# ── Init tests ─────────────────────────────────────────────────────────────

def test_init_raises_without_jina_key(mocker):
    """Kiểm tra ValueError khi không có JINA_API_KEY."""
    mock_s = MagicMock()
    mock_s.JINA_API_KEY = None
    mocker.patch("src.engine.ingestion.late_chunker.settings", mock_s)

    with pytest.raises(ValueError, match="JINA_API_KEY"):
        LateChunkingEmbedder()


def test_init_success():
    """Kiểm tra khởi tạo thành công khi có key."""
    embedder = LateChunkingEmbedder()
    assert embedder._api_key == "fake_jina_key"


# ── embed_chunks tests ─────────────────────────────────────────────────────

def test_embed_chunks_empty_returns_empty():
    """Kiểm tra input rỗng → trả về []."""
    embedder = LateChunkingEmbedder()
    result = embedder.embed_chunks([])
    assert result == []


def test_embed_chunks_single_batch():
    """Kiểm tra embed dưới batch_size (64) → 1 API call."""
    embedder = LateChunkingEmbedder()
    texts = [f"chunk {i}" for i in range(5)]

    mock_resp = _make_jina_response(5)
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        result = embedder.embed_chunks(texts)

    assert len(result) == 5
    assert len(result[0]) == 1024
    mock_client.post.assert_called_once()


def test_embed_chunks_multiple_batches():
    """Kiểm tra embed hơn 64 chunks → chia thành nhiều batch."""
    embedder = LateChunkingEmbedder()
    texts = [f"chunk {i}" for i in range(70)]  # 70 > BATCH_SIZE=64 → 2 batches

    def mock_post(url, json, headers):
        n = len(json["input"])
        return _make_jina_response(n)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = mock_post

        result = embedder.embed_chunks(texts)

    assert len(result) == 70
    assert mock_client.post.call_count == 2  # batch 1: 64, batch 2: 6


def test_embed_chunks_sends_correct_payload():
    """Kiểm tra payload gửi đúng các field Jina yêu cầu."""
    embedder = LateChunkingEmbedder()
    texts = ["hello", "world"]

    mock_resp = _make_jina_response(2)
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        embedder.embed_chunks(texts)

    call_kwargs = mock_client.post.call_args
    payload = call_kwargs.kwargs["json"]
    assert payload["late_chunking"] is True
    assert payload["dimensions"] == 1024
    assert payload["task"] == "retrieval.passage"
    assert payload["normalized"] is True
    assert payload["input"] == texts


def test_embed_chunks_sends_auth_header():
    """Kiểm tra Authorization header đúng format Bearer."""
    embedder = LateChunkingEmbedder()
    mock_resp = _make_jina_response(1)

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp

        embedder.embed_chunks(["test"])

    headers = mock_client.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer fake_jina_key"


def test_embed_chunks_raises_on_http_error():
    """Kiểm tra raise khi Jina API trả lỗi HTTP."""
    embedder = LateChunkingEmbedder()

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 401 Unauthorized")
        mock_client.post.return_value = mock_resp

        with pytest.raises(Exception, match="401"):
            embedder.embed_chunks(["test"])
