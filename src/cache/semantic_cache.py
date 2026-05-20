import hashlib
from typing import Optional

from qdrant_client.http.models import PointStruct

from src.core.config import settings
from src.core.database import db_instance
from src.core.logger import logger


class SemanticCache:
    """Qdrant-backed semantic cache: returns cached answers for semantically similar queries.

    On a cache hit (cosine similarity >= threshold), the LLM call is skipped entirely.
    Embeddings use the same BAAI/bge-m3 model already loaded in db_instance.
    """

    CACHE_COLLECTION = "semantic_cache"

    def __init__(self, similarity_threshold: Optional[float] = None):
        self.db = db_instance
        self.threshold = similarity_threshold if similarity_threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD
        self.db.get_or_create_collection(self.CACHE_COLLECTION)
        logger.info(f"[SemanticCache] Initialized (threshold={self.threshold}, collection={self.CACHE_COLLECTION})")

    def _embed(self, text: str) -> list[float]:
        """Embed a single text string using the shared embedding model."""
        vector = self.db.embedding_model.encode([text], normalize_embeddings=True)
        return vector[0].tolist()

    def _make_id(self, query: str) -> int:
        """Deterministic integer ID derived from query text (SHA-256 truncated to 63 bits)."""
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return int(digest, 16) % (2**63)

    def check_cache(self, query: str, collection_name: Optional[str] = None) -> Optional[dict]:
        """Embed query and search for a semantically similar cached answer.

        Filters by collection_name so cache entries from different videos don't cross-contaminate.
        Returns a dict {answer, sources, facts} if a hit is found, or None.
        """
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            vector = self._embed(query)

            query_filter = None
            if collection_name:
                query_filter = Filter(
                    must=[FieldCondition(key="collection_name", match=MatchValue(value=collection_name))]
                )

            results = self.db.client.query_points(
                collection_name=self.CACHE_COLLECTION,
                query=vector,
                query_filter=query_filter,
                limit=1,
                with_payload=True,
                score_threshold=self.threshold,
            )
            if results.points:
                hit = results.points[0]
                logger.info(
                    f"[SemanticCache] 🎯 HIT (score={hit.score:.4f}) col={collection_name} query='{query[:60]}'"
                )
                return {
                    "answer": hit.payload.get("answer"),
                    "sources": hit.payload.get("sources", []),
                    "facts": hit.payload.get("facts", [])
                }
        except Exception as e:
            logger.warning(f"[SemanticCache] check_cache error (non-fatal): {e}")
        return None

    def save_to_cache(self, query: str, answer: str, collection_name: Optional[str] = None, sources: list = None, facts: list = None) -> None:
        """Embed query and upsert (query_vector, answer, metadata) into the cache collection."""
        try:
            vector = self._embed(query)
            # ID bao gồm cả collection để tránh collision giữa các video
            key = f"{collection_name}::{query}" if collection_name else query
            point_id = self._make_id(key)
            payload = {
                "query": query,
                "collection_name": collection_name or "",
                "answer": answer,
                "sources": sources or [],
                "facts": facts or []
            }
            self.db.client.upsert(
                collection_name=self.CACHE_COLLECTION,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.info(f"[SemanticCache] ✅ SAVED col={collection_name} id={point_id} query='{query[:60]}'")
        except Exception as e:
            logger.warning(f"[SemanticCache] save_to_cache error (non-fatal): {e}")
