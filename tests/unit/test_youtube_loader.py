"""Tests for YouTubeLoader — URL parsing, text cleaning, metadata fallback, transcript errors."""

import pytest
from unittest.mock import MagicMock, patch

from src.engine.ingestion.youtube_loader import YouTubeLoader


@pytest.fixture
def loader():
    """YouTubeLoader instance — no heavy deps at init."""
    return YouTubeLoader()


# ── URL extraction tests ───────────────────────────────────────────────────

def test_extract_video_id_watch_url():
    """Kiểm tra URL dạng ?v=XXXX."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = YouTubeLoader.extract_video_id(url)
    assert result == "dQw4w9WgXcQ"


def test_extract_video_id_short_url():
    """Kiểm tra URL dạng youtu.be/XXXX."""
    url = "https://youtu.be/dQw4w9WgXcQ"
    result = YouTubeLoader.extract_video_id(url)
    assert result == "dQw4w9WgXcQ"


def test_extract_video_id_embed_url():
    """Kiểm tra URL dạng /embed/XXXX."""
    url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
    result = YouTubeLoader.extract_video_id(url)
    assert result == "dQw4w9WgXcQ"


def test_extract_video_id_shorts_url():
    """Kiểm tra URL dạng /shorts/XXXX."""
    url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
    result = YouTubeLoader.extract_video_id(url)
    assert result == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    """Kiểm tra URL không hợp lệ trả về None."""
    assert YouTubeLoader.extract_video_id("not-a-url") is None
    assert YouTubeLoader.extract_video_id("https://example.com/video") is None
    assert YouTubeLoader.extract_video_id("") is None


# ── Text cleaning tests ────────────────────────────────────────────────────

def test_clean_text_removes_unicode_artifacts():
    """Kiểm tra loại bỏ ký tự Unicode rác: \\xa0, \\u200b, \\ufeff.

    Note: \xa0 → space, \u200b → removed (zero-width, no space replacement), \ufeff → removed.
    """
    # \xa0 (non-breaking space) → regular space
    assert YouTubeLoader._clean_text("\xa0hello\xa0") == "hello"
    # \u200b (zero-width space) → removed entirely (no space injected)
    assert YouTubeLoader._clean_text("hello\u200bworld") == "helloworld"
    # \ufeff (BOM) → removed entirely
    assert YouTubeLoader._clean_text("\ufeffhello") == "hello"
    # Combined: \xa0 becomes space, extra spaces collapsed
    result = YouTubeLoader._clean_text("\xa0hello \xa0 world\ufeff")
    assert result == "hello world"


def test_clean_text_collapses_multiple_spaces():
    """Kiểm tra thu gọn khoảng trắng liên tiếp."""
    raw = "hello   world    test"
    result = YouTubeLoader._clean_text(raw)
    assert result == "hello world test"


def test_clean_text_converts_newline_to_space():
    """Kiểm tra newline bên trong caption được đổi thành space."""
    raw = "first line\nsecond line"
    result = YouTubeLoader._clean_text(raw)
    assert result == "first line second line"


# ── Metadata fallback test ─────────────────────────────────────────────────

def test_fetch_metadata_fallback_on_error(loader):
    """Kiểm tra fetch_metadata trả về minimal metadata khi pytubefix lỗi."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with patch("src.engine.ingestion.youtube_loader.YouTube") as mock_yt:
        mock_yt.side_effect = Exception("network error")
        result = loader.fetch_metadata(url)

    assert result["title"] == "Unknown Title"
    assert result["author"] == "Unknown Author"
    assert result["video_id"] == "dQw4w9WgXcQ"
    assert result["video_url"] == url
    assert result["length_sec"] == 0


# ── Transcript error handling tests ───────────────────────────────────────

def test_fetch_transcript_returns_empty_on_error(loader):
    """Kiểm tra fetch_transcript trả về [] khi API lỗi hoàn toàn."""
    with patch("src.engine.ingestion.youtube_loader.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch.side_effect = Exception("no transcript available")
        mock_instance.list.side_effect = Exception("no transcript available")
        mock_api.return_value = mock_instance

        result = loader.fetch_transcript("dQw4w9WgXcQ")

    assert result == []


# ── load_video_data validation test ───────────────────────────────────────

def test_load_video_data_raises_on_invalid_url(loader):
    """Kiểm tra load_video_data raise ValueError khi URL không hợp lệ."""
    with pytest.raises(ValueError, match="URL YouTube không hợp lệ"):
        loader.load_video_data("not-a-youtube-url")
