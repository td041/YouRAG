"""
Visual Frame RAG — extract frames from YouTube video, describe with a vision model,
embed, and upsert alongside text chunks.

Priority: Groq llama-3.2-11b-vision (primary) → GPT-4o-mini (fallback).
Both fail gracefully: visual RAG is skipped, text RAG still works.
"""
import os
import base64
import tempfile
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import settings
from src.core.logger import logger


_VISION_PROMPT = (
    "You are analyzing a single video frame for educational content indexing. "
    "Describe what you see concisely: text on slides, diagrams, code, mathematical "
    "formulas, charts, or concepts being demonstrated visually. "
    "Focus on educational content — what knowledge does this frame convey? "
    "Respond in the same language as text visible in the frame (Vietnamese or English). "
    "Keep description under 150 words. "
    "If the frame is blank, a loading screen, a face close-up with no educational content, "
    "or a scene transition, reply with exactly: NO_CONTENT"
)


class FrameExtractor:
    """Download YouTube video at lowest quality and extract JPEG frames at a fixed interval."""

    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds

    def extract(self, video_url: str) -> List[Tuple[float, bytes]]:
        """Return list of (timestamp_seconds, jpeg_bytes). Returns [] on any failure."""
        import shutil

        if not shutil.which("yt-dlp"):
            logger.error("[FrameExtractor] yt-dlp not installed — cannot extract frames")
            return []
        if not shutil.which("ffmpeg"):
            logger.error("[FrameExtractor] ffmpeg not installed — cannot extract frames")
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")

            logger.info("[FrameExtractor] Downloading video at lowest quality...")
            try:
                subprocess.run(
                    [
                        "yt-dlp",
                        "--format",
                        "worstvideo[ext=mp4]/worstvideo/worst[ext=mp4]/worst",
                        "--output", video_path,
                        "--quiet", "--no-warnings",
                        video_url,
                    ],
                    check=True,
                    timeout=600,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"[FrameExtractor] yt-dlp failed (rc={e.returncode})")
                return []
            except subprocess.TimeoutExpired:
                logger.error("[FrameExtractor] yt-dlp timed out after 600s")
                return []

            if not os.path.exists(video_path):
                logger.error("[FrameExtractor] Video file missing after yt-dlp")
                return []

            frames_dir = os.path.join(tmpdir, "frames")
            os.makedirs(frames_dir)

            logger.info(f"[FrameExtractor] Extracting 1 frame every {self.interval_seconds}s...")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i", video_path,
                        "-vf", f"fps=1/{self.interval_seconds},scale=640:-2",
                        "-q:v", "5",
                        os.path.join(frames_dir, "frame_%05d.jpg"),
                        "-loglevel", "error",
                    ],
                    check=True,
                    timeout=300,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"[FrameExtractor] ffmpeg failed (rc={e.returncode})")
                return []

            results: List[Tuple[float, bytes]] = []
            for i, path in enumerate(sorted(Path(frames_dir).glob("frame_*.jpg"))):
                ts = float(i * self.interval_seconds)
                results.append((ts, path.read_bytes()))

            logger.info(f"[FrameExtractor] {len(results)} frames extracted")
            return results


class VisualDescriber:
    """Describe a JPEG frame using vision models.

    Priority: Groq llama-3.2-11b-vision (primary, uses existing GROQ_API_KEY)
              → GPT-4o-mini (if OPENAI_API_KEY set)
    """

    def describe(self, image_bytes: bytes, timestamp: float) -> Optional[str]:
        """Return description string, or None if frame has no educational content."""
        text = (
            self._groq(image_bytes, timestamp)
            or self._openai(image_bytes, timestamp)
        )
        if not text or text.strip() == "NO_CONTENT":
            return None
        return text.strip()

    # ── Groq llama-3.2-11b-vision (primary — uses existing GROQ_API_KEY) ──
    def _groq(self, image_bytes: bytes, timestamp: float) -> Optional[str]:
        api_key = getattr(settings, "GROQ_API_KEY", None)
        if not api_key:
            return None
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            b64 = base64.b64encode(image_bytes).decode()
            ts = _fmt_ts(timestamp)
            resp = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{_VISION_PROMPT}\n\nFrame timestamp: {ts}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=250,
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.debug(f"[VisualDescriber] Groq vision failed: {e}")
            return None

    # ── GPT-4o-mini vision (optional fallback) ────────────────────────────
    def _openai(self, image_bytes: bytes, timestamp: float) -> Optional[str]:
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            return None
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            b64 = base64.b64encode(image_bytes).decode()
            ts = _fmt_ts(timestamp)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{_VISION_PROMPT}\n\nFrame timestamp: {ts}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=250,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.debug(f"[VisualDescriber] OpenAI failed: {e}")
            return None


class VisualRAGPipeline:
    """Full pipeline: frames → describe (parallel) → embed (BGE-M3) → upsert Qdrant."""

    def __init__(self, interval_seconds: int = 30, max_workers: int = 4):
        self.extractor = FrameExtractor(interval_seconds=interval_seconds)
        self.describer = VisualDescriber()
        self.max_workers = max_workers

    def run(
        self,
        video_url: str,
        metadata: Dict[str, Any],
        collection_name: str,
        chunk_index_start: int = 0,
    ) -> int:
        """Extract, describe, embed and upsert visual chunks. Returns count added."""
        from qdrant_client.http import models
        from src.core.database import db_instance

        frames = self.extractor.extract(video_url)
        if not frames:
            return 0

        logger.info(
            f"[VisualRAG] Describing {len(frames)} frames "
            f"(parallel, max_workers={self.max_workers})..."
        )

        described: List[Tuple[float, str]] = []

        def _describe_one(item: Tuple[float, bytes]) -> Tuple[float, Optional[str]]:
            ts, img = item
            return ts, self.describer.describe(img, ts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futs = {executor.submit(_describe_one, f): f[0] for f in frames}
            for fut in as_completed(futs):
                ts, text = fut.result()
                if text:
                    described.append((ts, text))

        described.sort(key=lambda x: x[0])
        logger.info(f"[VisualRAG] {len(described)}/{len(frames)} frames have educational content")

        if not described:
            logger.warning("[VisualRAG] All frames returned NO_CONTENT — check vision API keys")
            return 0

        # Prefix makes retrieval understand chunk type and timestamp
        docs = [
            f"[VISUAL at {_fmt_ts(ts)}] {text}"
            for ts, text in described
        ]

        embeddings = db_instance.embedding_model.encode(docs, show_progress_bar=False)

        video_id = metadata.get("video_id", "unknown")
        points = []
        for i, ((ts, _), doc, emb) in enumerate(zip(described, docs, embeddings)):
            payload: Dict[str, Any] = {
                "text": doc,
                "chunk_type": "visual",
                "video_id": video_id,
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "start_time": ts,
                "end_time": ts,
                "source": metadata.get("source", ""),
                "chunk_index": chunk_index_start + i,
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_visual_{i}")),
                    vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                    payload=payload,
                )
            )

        db_instance.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"[VisualRAG] ✅ {len(points)} visual chunks → [{collection_name}]")
        return len(points)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
