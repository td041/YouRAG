"""Unit tests for src/engine/ingestion/frame_extractor.py."""

import pytest
from unittest.mock import MagicMock
import subprocess


@pytest.fixture(autouse=True)
def mock_shutil(mocker):
    mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")


class TestFrameExtractor:
    def test_extract_returns_empty_when_ytdlp_missing(self, mocker):
        """yt-dlp không cài → trả về []."""
        mocker.patch("shutil.which", return_value=None)

        from src.engine.ingestion.frame_extractor import FrameExtractor
        fe = FrameExtractor()
        result = fe.extract("https://youtube.com/watch?v=test")
        assert result == []

    def test_extract_returns_empty_when_ytdlp_fails(self, mocker):
        """yt-dlp CalledProcessError → trả về []."""
        mocker.patch("shutil.which", return_value="/usr/bin/tool")
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "yt-dlp"),
        )

        from src.engine.ingestion.frame_extractor import FrameExtractor
        fe = FrameExtractor()
        result = fe.extract("https://youtube.com/watch?v=test")
        assert result == []

    def test_extract_returns_empty_on_timeout(self, mocker):
        """yt-dlp timeout → trả về []."""
        mocker.patch("shutil.which", return_value="/usr/bin/tool")
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("yt-dlp", 600),
        )

        from src.engine.ingestion.frame_extractor import FrameExtractor
        fe = FrameExtractor()
        result = fe.extract("https://youtube.com/watch?v=test")
        assert result == []

    def test_interval_seconds_default(self):
        """interval_seconds mặc định là 30."""
        from src.engine.ingestion.frame_extractor import FrameExtractor
        fe = FrameExtractor()
        assert fe.interval_seconds == 30

    def test_interval_seconds_custom(self):
        """interval_seconds có thể truyền custom."""
        from src.engine.ingestion.frame_extractor import FrameExtractor
        fe = FrameExtractor(interval_seconds=60)
        assert fe.interval_seconds == 60


class TestVisualDescriber:
    def test_describe_returns_none_when_no_api_keys(self, mocker):
        """Không có API key → trả về None."""
        mocker.patch("src.engine.ingestion.frame_extractor.settings.GROQ_API_KEY", None)
        mocker.patch("src.engine.ingestion.frame_extractor.settings.OPENAI_API_KEY", None)

        from src.engine.ingestion.frame_extractor import VisualDescriber
        vd = VisualDescriber()
        result = vd.describe(b"fake_jpeg", 0.0)
        assert result is None

    def test_describe_returns_none_on_no_content(self, mocker):
        """LLM trả về NO_CONTENT → trả về None."""
        mock_api_key = MagicMock()
        mock_api_key.get_secret_value.return_value = "test-key"
        mocker.patch("src.engine.ingestion.frame_extractor.settings.GROQ_API_KEY", mock_api_key)

        mock_groq = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "NO_CONTENT"
        mock_groq.return_value.chat.completions.create.return_value = mock_resp
        mocker.patch("src.engine.ingestion.frame_extractor.Groq", mock_groq, create=True)

        from src.engine.ingestion.frame_extractor import VisualDescriber
        vd = VisualDescriber()
        mocker.patch.object(vd, "_groq", return_value="NO_CONTENT")
        mocker.patch.object(vd, "_openai", return_value=None)
        result = vd.describe(b"fake_jpeg", 5.0)
        assert result is None

    def test_describe_strips_whitespace(self, mocker):
        """describe() strips whitespace từ kết quả LLM."""
        from src.engine.ingestion.frame_extractor import VisualDescriber
        vd = VisualDescriber()
        # Patch _groq để trả về text có whitespace, _openai trả về None
        mocker.patch.object(vd, "_groq", return_value="  A slide with text  ")
        mocker.patch.object(vd, "_openai", return_value=None)

        result = vd.describe(b"fake_jpeg", 10.0)
        assert result == "A slide with text"
