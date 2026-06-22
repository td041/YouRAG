"""
Contextual Compressor — trích chỉ những câu liên quan từ chunk trước khi pass LLM.

Dùng bge-m3 (đã load sẵn) để score từng câu vs query.
Giữ câu nào có cosine similarity > threshold, bỏ noise.
Kết quả: context sạch hơn → LLM faithful hơn, ít hallucinate hơn.
"""

import re
from typing import Any, Dict, List

from src.core.database import db_instance
from src.core.logger import logger

_SIMILARITY_THRESHOLD = 0.45  # câu nào dưới ngưỡng này bị bỏ
_MIN_SENTENCE_CHARS = 20       # câu quá ngắn bỏ qua
_MIN_SENTENCES_TO_COMPRESS = 3  # chunk < 3 câu thì giữ nguyên


def _split_sentences(text: str) -> List[str]:
    """Tách đoạn văn thành câu. Xử lý cả dấu chấm tiếng Việt."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_SENTENCE_CHARS]


def _cosine_sim(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class ContextualCompressor:
    """Lọc câu không liên quan ra khỏi mỗi chunk trước khi pass LLM.

    Dùng embedding similarity (bge-m3) để score từng câu vs query.
    Chunk ngắn (< 3 câu) được giữ nguyên để tránh mất thông tin.
    """

    def __init__(self, threshold: float = _SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self._db = db_instance

    def compress(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress từng chunk — giữ nguyên structure, chỉ rút ngắn content."""
        if not chunks:
            return chunks

        try:
            query_vec = self._db.embedding_model.encode([query])[0]
        except Exception as e:
            logger.warning(f"[Compressor] encode query failed: {e}")
            return chunks

        compressed = []
        total_before = total_after = 0

        for chunk in chunks:
            content = chunk.get("content", "")
            sentences = _split_sentences(content)

            if len(sentences) < _MIN_SENTENCES_TO_COMPRESS:
                compressed.append(chunk)
                total_before += len(content)
                total_after += len(content)
                continue

            try:
                sent_vecs = self._db.embedding_model.encode(sentences)
                relevant = [
                    s for s, vec in zip(sentences, sent_vecs)
                    if _cosine_sim(query_vec, vec) >= self.threshold
                ]
            except Exception:
                relevant = []

            # Nếu lọc hết thì giữ nguyên (an toàn hơn là mất context)
            new_content = " ".join(relevant) if relevant else content

            total_before += len(content)
            total_after += len(new_content)

            new_chunk = chunk.copy()
            new_chunk["content"] = new_content
            compressed.append(new_chunk)

        reduction = (1 - total_after / max(total_before, 1)) * 100
        logger.info(f"[Compressor] {len(chunks)} chunks: {total_before}→{total_after} chars ({reduction:.0f}% reduced)")
        return compressed
