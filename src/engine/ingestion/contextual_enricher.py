"""
Contextual Chunking — Anthropic (Sep 2024)
https://www.anthropic.com/news/contextual-retrieval

Kỹ thuật:
  Với mỗi chunk, gửi (full_document + chunk) lên LLM nhỏ → nhận lại 1-2 câu
  giải thích chunk này nằm ở đâu trong document → prepend vào chunk trước khi embed.

Kết quả reported: +49% recall@20 so với naive chunking.

Thiết kế triển khai:
  1. File-based cache (hash của video_id + chunk_index) → không gọi LLM lại khi re-ingest
  2. Parallel LLM calls (ThreadPoolExecutor) → tốc độ scale tuyến tính với worker
  3. Graceful degrade: nếu LLM lỗi → giữ chunk gốc, log warning, tiếp tục
  4. Full transcript bị truncate nếu quá dài → tránh vượt context window LLM
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from src.core.logger import logger
from src.core.redis_client import get_redis
from src.engine.generation.llm_client import LLMClient

_CTX_REDIS_TTL = 60 * 60 * 24 * 30  # 30 days
_CTX_KEY_PREFIX = "ctx_cache:"


# ── Prompt chính xác theo paper Anthropic ────────────────────────────────────
CONTEXT_PROMPT_TEMPLATE = """\
<document>
{full_text}
</document>

Here is a chunk from the above document:
<chunk>
{chunk_text}
</chunk>

Provide a short context (1-2 sentences) to situate this chunk within the overall document \
for the purposes of improving search retrieval. Be specific about:
- WHAT topic or action this chunk covers
- WHERE it appears in the document flow (beginning/middle/end, after what topic)

Reply with ONLY the context sentence(s). No preamble, no explanation."""

# Nếu transcript quá dài (>vượt LLM context window), truncate từ 2 phía để giữ đầu + cuối
MAX_FULL_TEXT_CHARS = 12_000  # ~3000 tokens, an toàn với Llama3-8b (8k context)


class ContextualEnricher:
    """Implement Anthropic Contextual Retrieval cho YouTube transcript chunks.
    
    Sử dụng:
        enricher = ContextualEnricher()
        enriched_chunks = enricher.enrich(full_transcript_text, chunks)
    """

    def __init__(
        self,
        max_workers: int = 5,           # Tăng lên 5 workers để song song nhanh hơn
        cache_dir: str = ".cache/contextual",
        llm_client: Optional[LLMClient] = None,
    ):
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._prune_disk_cache(max_age_days=30)

        # Khởi tạo LLM client (lazy — chỉ khi cần)
        self._llm: Optional[LLMClient] = llm_client
        self._llm_init_error: Optional[str] = None

        try:
            if self._llm is None:
                self._llm = LLMClient()
        except Exception as e:
            self._llm_init_error = str(e)
            logger.warning(f"⚠️  ContextualEnricher: LLM không khả dụng → sẽ bỏ qua enrichment. Lý do: {e}")

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _prune_disk_cache(self, max_age_days: int = 30) -> None:
        """Delete local cache files older than max_age_days."""
        try:
            cutoff = time.time() - max_age_days * 86400
            removed = 0
            for fname in os.listdir(self.cache_dir):
                fpath = os.path.join(self.cache_dir, fname)
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    removed += 1
            if removed:
                logger.info(f"[ContextualCache] Pruned {removed} stale files (>{max_age_days}d old)")
        except Exception as e:
            logger.warning(f"[ContextualCache] Prune failed: {e}")

    def _cache_key(self, video_id: str, chunk_index: int, chunk_text: str) -> str:
        """Hash duy nhất cho mỗi (video, chunk) → dùng làm tên file cache."""
        raw = f"{video_id}|{chunk_index}|{chunk_text}"
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _read_cache(self, key: str) -> Optional[str]:
        # 1. Try Redis
        r = get_redis()
        if r:
            val = r.get(f"{_CTX_KEY_PREFIX}{key}")
            if val is not None:
                return val

        # 2. Fallback: local file
        path = self._cache_path(key)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)["context"]
            except Exception as e:
                logger.debug(f"[ContextualEnricher] Local cache read failed ({path}): {e}")
        return None

    def _write_cache(self, key: str, context: str):
        # 1. Write to Redis (primary)
        r = get_redis()
        if r:
            try:
                r.setex(f"{_CTX_KEY_PREFIX}{key}", _CTX_REDIS_TTL, context)
                return
            except Exception as e:
                logger.warning(f"Redis write failed, falling back to file: {e}")

        # 2. Fallback: local file
        path = self._cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"context": context}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    # ── Full text preparation ─────────────────────────────────────────────────

    @staticmethod
    def _prepare_full_text(full_text: str) -> str:
        """Truncate full transcript nếu quá dài, giữ đầu + cuối (quan trọng nhất)."""
        if len(full_text) <= MAX_FULL_TEXT_CHARS:
            return full_text
        
        # Lấy 60% đầu + 40% cuối → giữ intro + conclusion của video
        head_size = int(MAX_FULL_TEXT_CHARS * 0.6)
        tail_size = MAX_FULL_TEXT_CHARS - head_size
        truncated = (
            full_text[:head_size]
            + "\n\n[... transcript truncated for context window ...]\n\n"
            + full_text[-tail_size:]
        )
        logger.debug(f"Transcript truncated: {len(full_text)} → {len(truncated)} chars")
        return truncated

    # ── Single chunk enrichment ───────────────────────────────────────────────

    def _enrich_single(
        self,
        full_text: str,
        chunk: Dict,
        chunk_index: int,
        video_id: str,
    ) -> Dict:
        """Sinh context cho 1 chunk. Đọc cache trước, gọi LLM nếu cache miss."""
        chunk_text = chunk["content"]
        cache_key = self._cache_key(video_id, chunk_index, chunk_text)

        # 1. Cache hit → trả về ngay
        cached = self._read_cache(cache_key)
        if cached:
            logger.debug(f"  [cache hit] chunk {chunk_index}")
            enriched_content = f"{cached}\n\n{chunk_text}"
            return {**chunk, "content": enriched_content, "metadata": {**chunk["metadata"], "has_context": True}}

        # 2. Cache miss → gọi LLM
        prompt = CONTEXT_PROMPT_TEMPLATE.format(
            full_text=full_text,
            chunk_text=chunk_text,
        )
        try:
            context = self._llm.chat_complete(
                prompt=prompt,
                system="You are a document analysis assistant. Be concise and factual.",
                max_tokens=120,
                temperature=0.0,
                model="llama-3.1-8b-instant" # Sử dụng model nhỏ nhất để cực nhanh cho enrichment
            )
            self._write_cache(cache_key, context)
            logger.debug(f"  [LLM] chunk {chunk_index}: \"{context[:80]}...\"")

            enriched_content = f"{context}\n\n{chunk_text}"
            return {
                **chunk,
                "content": enriched_content,
                "metadata": {**chunk["metadata"], "has_context": True, "context_snippet": context[:100]},
            }

        except Exception as e:
            logger.warning(f"  [LLM error] chunk {chunk_index}: {e} → dùng chunk gốc")
            return {**chunk, "metadata": {**chunk["metadata"], "has_context": False}}

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(self, full_transcript_text: str, chunks: List[Dict]) -> List[Dict]:
        """Enrich toàn bộ chunks với contextual prefix.

        Args:
            full_transcript_text: Toàn bộ text transcript (đã compile, chưa chunk)
            chunks: Output từ SemanticChunker

        Returns:
            List chunks đã được prepend context. Nếu LLM không khả dụng,
            trả về chunks gốc (graceful degrade).
        """
        if not chunks:
            return chunks

        # Nếu LLM không khả dụng → bỏ qua enrichment hoàn toàn
        if self._llm is None:
            logger.warning("LLM không khả dụng → bỏ qua Contextual Enrichment")
            return chunks

        video_id = chunks[0]["metadata"].get("video_id", "unknown")
        full_text = self._prepare_full_text(full_transcript_text)

        logger.info(f"🧠 Contextual Enrichment: {len(chunks)} chunks | video={video_id}")
        logger.info(f"   Full text: {len(full_text)} chars | LLM: {self._llm.provider}/{self._llm.model}")
        logger.info(f"   Parallel workers: {self.max_workers} | Cache: {self.cache_dir}")

        t_start = time.perf_counter()
        results = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._enrich_single, full_text, chunk, i, video_id
                ): i
                for i, chunk in enumerate(chunks)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Enrich chunk {idx} thất bại: {e}")
                    results[idx] = chunks[idx]
                completed += 1
                if completed % 5 == 0 or completed == len(chunks):
                    logger.info(f"   Progress: {completed}/{len(chunks)} chunks enriched")

        elapsed = time.perf_counter() - t_start
        cache_hits = sum(1 for c in results if c and c.get("metadata", {}).get("has_context"))
        
        logger.info(
            f"✅ Contextual Enrichment hoàn tất: {elapsed:.2f}s | "
            f"cache_hits={cache_hits}/{len(chunks)}"
        )
        return results
