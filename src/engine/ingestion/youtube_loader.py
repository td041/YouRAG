import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

from youtube_transcript_api import YouTubeTranscriptApi
from pytubefix import YouTube

from src.core.config import settings
from src.core.logger import logger


class YouTubeLoader:
    """Lớp chuyên dụng cào Transcript (Phụ đề) và Metadata từ YouTube.
    
    Tối ưu:
    - Parallel fetch: metadata + transcript chạy đồng thời → tiết kiệm ~50% thời gian
    - Text cleaning: loại bỏ ký tự Unicode rác (\xa0, zero-width...) từ auto-generated captions
    - Fallback metadata: nếu pytubefix lỗi, fallback về dữ liệu transcript API
    """

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Tách ID video từ mọi dạng URL YouTube hợp lệ (watch, youtu.be, embed)"""
        patterns = [
            r"(?:v=)([0-9A-Za-z_-]{11})",       # ?v=XXXX
            r"youtu\.be\/([0-9A-Za-z_-]{11})",    # youtu.be/XXXX
            r"embed\/([0-9A-Za-z_-]{11})",         # /embed/XXXX
            r"shorts\/([0-9A-Za-z_-]{11})",        # /shorts/XXXX
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Làm sạch text transcript:
        - Chuẩn hóa Unicode (NFC)
        - Xóa ký tự zero-width, \xa0 (non-breaking space)
        - Thu gọn khoảng trắng thừa
        - Loại bỏ newline thừa bên trong 1 dòng phụ đề
        """
        # Chuẩn hóa Unicode NFC
        text = unicodedata.normalize("NFC", text)
        # Thay \xa0 và các no-break space → space thường
        text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
        # Newline bên trong 1 caption line → space (auto-generated captions hay có)
        text = text.replace("\n", " ")
        # Thu gọn khoảng trắng liên tiếp
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def fetch_metadata(self, url: str) -> Dict:
        """Kéo Metadata từ pytubefix. Nếu lỗi, trả về minimal metadata (không crash)."""
        try:
            yt = YouTube(url)
            meta = {
                "title": yt.title or "Unknown Title",
                "author": yt.author or "Unknown Author",
                "length_sec": yt.length or 0,
                "publish_date": str(yt.publish_date) if yt.publish_date else "Unknown",
                "views": yt.views or 0,
                "video_url": url,
                "video_id": self.extract_video_id(url),
            }
            logger.info(f"Metadata: '{meta['title']}' | {meta['length_sec']}s | {meta['views']:,} views")
            return meta
        except Exception as e:
            logger.warning(f"pytubefix không lấy được metadata (sẽ dùng fallback): {e}")
            return {
                "title": "Unknown Title",
                "author": "Unknown Author",
                "length_sec": 0,
                "publish_date": "Unknown",
                "views": 0,
                "video_url": url,
                "video_id": self.extract_video_id(url),
            }

    def fetch_transcript(self, video_id: str, languages: List[str] = ["vi", "en"]) -> List[Dict]:
        """Tải + làm sạch Transcript. Ưu tiên vi → en, nếu không có → thử auto-generated."""
        try:
            logger.info(f"Đang kéo Transcript cho video_id: {video_id}...")
            api = YouTubeTranscriptApi()

            # Thử lấy transcript theo ngôn ngữ ưu tiên
            try:
                fetched = api.fetch(video_id, languages=languages)
            except Exception:
                # Fallback: lấy bất kỳ transcript nào có sẵn (kể cả auto-generated)
                logger.warning("Không tìm thấy vi/en → thử lấy transcript tự động bất kỳ...")
                transcript_list = api.list(video_id)
                fetched = next(iter(transcript_list)).fetch()

            # Convert + làm sạch text ngay tại đây
            transcript = [
                {
                    "text": self._clean_text(s.text),
                    "start": round(s.start, 3),
                    "duration": round(s.duration, 3),
                }
                for s in fetched
                if s.text.strip()  # Bỏ qua các dòng trống
            ]

            logger.info(f"✅ Transcript: {len(transcript)} dòng phụ đề (đã làm sạch)")
            return transcript

        except Exception as e:
            logger.error(f"Không thể lấy Transcript của {video_id}. Nguyên nhân: {e}")
            return []

    def _transcribe_with_whisper(self, url: str) -> List[Dict]:
        """Fallback: tải audio và tự chuyển giọng nói → văn bản bằng faster-whisper (local).

        Dùng khi video không có phụ đề (CC) trên YouTube.
        Model 'base' (~145MB) tự download lần đầu, chạy được trên CPU và GPU.
        """
        try:
            from faster_whisper import WhisperModel  # lazy import
        except ImportError:
            raise RuntimeError(
                "faster-whisper chưa được cài. Chạy: poetry add faster-whisper"
            )

        import tempfile

        logger.info("🎙️ [Whisper] Không có phụ đề → đang tải audio để tự nhận dạng...")

        with tempfile.TemporaryDirectory() as tmpdir:
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).order_by("abr").last()
            if not audio_stream:
                raise ValueError("Không tìm thấy audio stream cho video này.")

            audio_path = audio_stream.download(output_path=tmpdir, filename="audio")
            logger.info(f"🎙️ [Whisper] Audio đã tải ({audio_stream.abr}), đang transcribe...")

            device = "cuda" if settings.DEVICE == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            model = WhisperModel("base", device=device, compute_type=compute_type)
            segments, info = model.transcribe(audio_path, beam_size=5)

            logger.info(
                f"🎙️ [Whisper] Ngôn ngữ phát hiện: {info.language} "
                f"(confidence: {info.language_probability:.2f})"
            )

            transcript = []
            for seg in segments:
                text = self._clean_text(seg.text)
                if text:
                    transcript.append({
                        "text": text,
                        "start": round(seg.start, 3),
                        "duration": round(seg.end - seg.start, 3),
                    })

        logger.info(f"✅ [Whisper] Transcribe xong: {len(transcript)} đoạn")
        return transcript

    def load_video_data(self, url: str) -> Dict:
        """Entry point: Fetch metadata + transcript SONG SONG để tối ưu thời gian.

        Nếu YouTube không có phụ đề → tự động fallback sang faster-whisper (local STT).
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"URL YouTube không hợp lệ: {url}")

        logger.info(f"⚡ Parallel fetch metadata + transcript cho: {video_id}")

        # Chạy song song 2 tác vụ I/O-bound, timeout 60s mỗi task
        metadata, raw_transcript = None, None
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_meta = executor.submit(self.fetch_metadata, url)
            future_transcript = executor.submit(self.fetch_transcript, video_id)

            metadata = future_meta.result(timeout=60)
            raw_transcript = future_transcript.result(timeout=60)

        # Whisper fallback khi YouTube không có phụ đề
        if not raw_transcript:
            logger.warning(
                f"⚠️ Video {video_id} không có phụ đề. Thử Whisper fallback..."
            )
            try:
                raw_transcript = self._transcribe_with_whisper(url)
            except Exception as whisper_err:
                logger.error(f"❌ Whisper fallback thất bại: {whisper_err}")
                raise ValueError(
                    f"Không tìm thấy transcript cho video {video_id} và Whisper cũng thất bại: "
                    f"{whisper_err}. Đảm bảo faster-whisper đã được cài đặt."
                )

        if not raw_transcript:
            raise ValueError(
                f"Không thể lấy transcript cho video {video_id} bằng bất kỳ phương thức nào."
            )

        return {
            "metadata": metadata,
            "transcript": raw_transcript,  # List[Dict(text, start, duration)]
        }
