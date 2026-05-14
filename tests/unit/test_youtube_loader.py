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


# ── Success path tests ────────────────────────────────────────────────────

def test_fetch_metadata_success(loader):
    """Kiểm tra fetch_metadata lấy đúng thông tin khi pytubefix hoạt động."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with patch("src.engine.ingestion.youtube_loader.YouTube") as mock_yt:
        mock_instance = MagicMock()
        mock_instance.title = "Test Video"
        mock_instance.author = "Test Author"
        mock_instance.length = 300
        mock_instance.publish_date = "2024-01-01"
        mock_instance.views = 1000
        mock_yt.return_value = mock_instance

        result = loader.fetch_metadata(url)

    assert result["title"] == "Test Video"
    assert result["author"] == "Test Author"
    assert result["length_sec"] == 300
    assert result["views"] == 1000


def test_fetch_transcript_success(loader):
    """Kiểm tra fetch_transcript lấy + làm sạch transcript thành công."""
    with patch("src.engine.ingestion.youtube_loader.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()

        # Mock transcript snippets
        s1 = MagicMock()
        s1.text = "Hello world"
        s1.start = 0.0
        s1.duration = 1.5

        s2 = MagicMock()
        s2.text = "Test\xa0content"  # Has non-breaking space
        s2.start = 1.5
        s2.duration = 2.0

        mock_instance.fetch.return_value = [s1, s2]
        mock_api.return_value = mock_instance

        result = loader.fetch_transcript("dQw4w9WgXcQ")

    assert len(result) == 2
    assert result[0]["text"] == "Hello world"
    assert result[1]["text"] == "Test content"  # \xa0 cleaned


def test_fetch_transcript_fallback_to_auto_generated(loader):
    """Kiểm tra fallback lấy transcript auto-generated khi vi/en không có."""
    with patch("src.engine.ingestion.youtube_loader.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()

        # fetch() raises for vi/en
        mock_instance.fetch.side_effect = Exception("No vi/en")

        # Fallback: list().fetch() trả về transcript auto
        s1 = MagicMock()
        s1.text = "Auto generated"
        s1.start = 0.0
        s1.duration = 1.0

        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = [s1]
        mock_instance.list.return_value = iter([mock_transcript])
        mock_api.return_value = mock_instance

        result = loader.fetch_transcript("dQw4w9WgXcQ")

    assert len(result) == 1
    assert result[0]["text"] == "Auto generated"


def test_load_video_data_success(loader):
    """Kiểm tra load_video_data chạy song song metadata + transcript."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with patch.object(loader, "fetch_metadata") as mock_meta, \
         patch.object(loader, "fetch_transcript") as mock_trans:
        mock_meta.return_value = {"title": "Vid", "video_id": "dQw4w9WgXcQ"}
        mock_trans.return_value = [{"text": "Hello", "start": 0, "duration": 1}]

        result = loader.load_video_data(url)

    assert result["metadata"]["title"] == "Vid"
    assert len(result["transcript"]) == 1


def test_load_video_data_raises_on_empty_transcript(loader):
    """Kiểm tra raise ValueError khi transcript trống."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with patch.object(loader, "fetch_metadata") as mock_meta, \
         patch.object(loader, "fetch_transcript") as mock_trans:
        mock_meta.return_value = {"title": "Vid"}
        mock_trans.return_value = []  # Empty transcript

        with pytest.raises(ValueError, match="Không tìm thấy Transcript"):
            loader.load_video_data(url)

