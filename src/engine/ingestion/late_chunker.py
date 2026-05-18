"""Late Chunking Embedder — Jina AI jina-embeddings-v3.

Thay vì embed từng chunk độc lập, Late Chunking embed toàn bộ document một lần,
rồi pool token embeddings theo ranh giới chunk. Kết quả: mỗi chunk embedding
mang đầy đủ ngữ cảnh của cả video → tăng retrieval quality đáng kể với video dài.

Yêu cầu: JINA_API_KEY trong .env
Vector dim: 1024 (tương thích với BAAI/bge-m3, không cần đổi Qdrant collection)
"""

from typing import List

import httpx

from src.core.config import settings
from src.core.logger import logger

_JINA_URL = "https://api.jina.ai/v1/embeddings"
_MODEL = "jina-embeddings-v3"
_DIMENSIONS = 1024  # Giống BAAI/bge-m3 → tương thích collection hiện tại
_BATCH_SIZE = 64    # Jina khuyến nghị batch tối đa ~128 inputs


class LateChunkingEmbedder:
    """Embed danh sách chunk với full-document context qua Jina Late Chunking API.

    Late Chunking hoạt động:
    1. Jina nhận tất cả chunk của 1 video dưới dạng mảng input
    2. Model embed toàn bộ chuỗi token ghép lại (không tách biệt từng chunk)
    3. Pooling token embeddings theo ranh giới → 1 vector/chunk, nhưng context-aware
    """

    def __init__(self) -> None:
        if not settings.JINA_API_KEY:
            raise ValueError(
                "JINA_API_KEY chưa được cấu hình. "
                "Đăng ký miễn phí tại https://jina.ai và thêm vào .env"
            )
        self._api_key = settings.JINA_API_KEY.get_secret_value()

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Embed tất cả chunks của 1 video với full-document context.

        Args:
            texts: Danh sách nội dung chunk (theo thứ tự xuất hiện trong video).

        Returns:
            List of 1024-dim float vectors, tương thích với Qdrant collection hiện tại.
        """
        if not texts:
            return []

        logger.info(
            f"[LateChunking] Embedding {len(texts)} chunks qua Jina API "
            f"(model={_MODEL}, dim={_DIMENSIONS})..."
        )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        all_embeddings: List[List[float]] = []

        # Chia batch để tránh vượt giới hạn payload Jina
        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start: batch_start + _BATCH_SIZE]
            payload = {
                "model": _MODEL,
                "input": batch,
                "late_chunking": True,
                "dimensions": _DIMENSIONS,
                "task": "retrieval.passage",
                "normalized": True,
            }

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(_JINA_URL, json=payload, headers=headers)
                resp.raise_for_status()

            data = resp.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

            logger.info(
                f"[LateChunking] Batch {batch_start // _BATCH_SIZE + 1}: "
                f"{len(batch_embeddings)} embeddings ✓"
            )

        logger.info(
            f"[LateChunking] ✅ Hoàn tất: {len(all_embeddings)} context-aware embeddings"
        )
        return all_embeddings
