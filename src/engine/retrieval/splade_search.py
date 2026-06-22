"""
SPLADE Sparse Retriever — thay thế BM25 với neural sparse embeddings.

Model: naver/efficient-splade-VI-BT-large (bilingual training, EN + VI)
- Doc encoder: naver/efficient-splade-VI-BT-large-doc
- Query encoder: naver/efficient-splade-VI-BT-large-query

Interface giống hệt SparseRetriever (BM25) — drop-in swap trong HybridRetriever.
Model được lazy-load lần đầu tiên search, sau đó cache lại trong bộ nhớ.
"""

import threading
from typing import Any, Dict, List, Optional

from src.core.database import db_instance
from src.core.logger import logger

_SPLADE_MAX_DOCS = 2000
_DOC_MODEL = "naver/efficient-splade-VI-BT-large-doc"
_QUERY_MODEL = "naver/efficient-splade-VI-BT-large-query"
_BATCH_SIZE = 32


class SpladeRetriever:
    """Sparse retrieval dùng SPLADE neural sparse embeddings.

    Thay thế BM25, đặc biệt hiệu quả hơn với tiếng Việt do model được train bilingual.
    Vectors được cache in-memory per collection (giống BM25), rebuild khi server restart.
    Thread-safe: double-checked locking cho model load và index build.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.db = db_instance
        self._doc_vectors_cache: Dict[str, List[Dict[int, float]]] = {}
        self._doc_mappings: Dict[str, List[Dict]] = {}
        self._models_loaded = False
        self._doc_tokenizer = None
        self._doc_model = None
        self._query_tokenizer = None
        self._query_model = None
        self._device: Optional[str] = None
        self._model_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._index_locks: Dict[str, threading.Lock] = {}
        self._index_locks_lock = threading.Lock()

    def _get_index_lock(self, collection_name: str) -> threading.Lock:
        with self._index_locks_lock:
            if collection_name not in self._index_locks:
                self._index_locks[collection_name] = threading.Lock()
            return self._index_locks[collection_name]

    def _load_models(self) -> bool:
        """Lazy-load SPLADE models. Thread-safe via double-checked locking."""
        if self._models_loaded:
            return True
        with self._model_lock:
            if self._models_loaded:  # re-check inside lock
                return True
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"⚙️ Loading SPLADE models on {self._device}...")

                self._doc_tokenizer = AutoTokenizer.from_pretrained(_DOC_MODEL)
                self._doc_model = (
                    AutoModelForMaskedLM.from_pretrained(_DOC_MODEL, low_cpu_mem_usage=False)
                    .to(self._device)
                    .eval()
                )
                self._query_tokenizer = AutoTokenizer.from_pretrained(_QUERY_MODEL)
                self._query_model = (
                    AutoModelForMaskedLM.from_pretrained(_QUERY_MODEL, low_cpu_mem_usage=False)
                    .to(self._device)
                    .eval()
                )
                self._models_loaded = True
                logger.info("✅ SPLADE models loaded")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load SPLADE models: {e}")
                return False

    def _encode_batch(self, texts: List[str], tokenizer, model) -> List[Dict[int, float]]:
        """Encode batch of texts → list of sparse dicts {token_id: weight}."""
        import torch

        results: List[Dict[int, float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                logits = model(**inputs).logits  # [B, seq_len, vocab]

            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, seq_len, 1]
            # SPLADE aggregation: max over seq positions of log(1 + relu(x))
            vecs = torch.max(
                torch.log(1 + torch.relu(logits)) * mask, dim=1
            ).values  # [B, vocab]

            for vec in vecs:
                nonzero_ids = vec.nonzero().squeeze(1).tolist()
                results.append({idx: vec[idx].item() for idx in nonzero_ids})

        return results

    # Sentinel value written to _doc_vectors_cache on build failure so that
    # subsequent search() calls skip the expensive retry immediately.
    _FAILED = object()

    def _build_or_get_vectors(self, collection_name: str) -> bool:
        """Fetch all docs, encode với SPLADE doc model, cache in-memory. Thread-safe."""
        cached = self._doc_vectors_cache.get(collection_name)
        if cached is not None:
            return cached is not self._FAILED

        lock = self._get_index_lock(collection_name)
        with lock:
            cached = self._doc_vectors_cache.get(collection_name)
            if cached is not None:  # re-check inside lock
                return cached is not self._FAILED
            if not self._load_models():
                return False

            logger.info(f"⚙️ Building SPLADE index for '{collection_name}'...")
            try:
                records: list = []
                offset = None
                while len(records) < _SPLADE_MAX_DOCS:
                    result, next_offset = self.db.client.scroll(
                        collection_name=collection_name,
                        limit=min(1000, _SPLADE_MAX_DOCS - len(records)),
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    records.extend(result)
                    if next_offset is None:
                        break
                    offset = next_offset

                if len(records) >= _SPLADE_MAX_DOCS:
                    logger.warning(f"[SPLADE] '{collection_name}' hit {_SPLADE_MAX_DOCS} doc cap")

                if not records:
                    raise ValueError("Collection empty")

                texts: List[str] = []
                mappings: List[Dict] = []
                for rec in records:
                    payload = rec.payload or {}
                    text = payload.get("text", "")
                    texts.append(text)
                    meta = {k: v for k, v in payload.items() if k != "text"}
                    mappings.append({"id": rec.id, "content": text, "metadata": meta})

                doc_vectors = self._encode_batch(texts, self._doc_tokenizer, self._doc_model)
                self._doc_vectors_cache[collection_name] = doc_vectors
                self._doc_mappings[collection_name] = mappings
                logger.info(f"✅ SPLADE index built: {len(records)} docs")
                return True

            except Exception as e:
                logger.error(f"❌ SPLADE index build failed for '{collection_name}': {e}")
                self._doc_vectors_cache[collection_name] = self._FAILED
                return False

    def clear_cache(self, collection_name: str) -> None:
        """Xoá cache của collection — gọi sau khi re-ingest video."""
        self._doc_vectors_cache.pop(collection_name, None)
        self._doc_mappings.pop(collection_name, None)
        logger.info(f"🗑️ SPLADE cache cleared for '{collection_name}'")

    def search(self, query: str, collection_name: str) -> List[Dict[str, Any]]:
        """SPLADE sparse search. Same interface as SparseRetriever (BM25)."""
        logger.info(f"🔍 SPLADE Search: '{query[:60]}' -> [{collection_name}]")
        try:
            if not self._build_or_get_vectors(collection_name):
                return []

            with self._encode_lock:
                query_vecs = self._encode_batch(
                    [query], self._query_tokenizer, self._query_model
                )
            query_vec = query_vecs[0]

            doc_vectors = self._doc_vectors_cache[collection_name]
            mappings = self._doc_mappings[collection_name]

            # Dot product: Σ query_weight[k] * doc_weight[k]
            scored: List[tuple] = []
            for i, doc_vec in enumerate(doc_vectors):
                score = sum(query_vec.get(k, 0.0) * v for k, v in doc_vec.items())
                if score > 0:
                    scored.append((i, score))

            scored.sort(key=lambda x: x[1], reverse=True)

            results: List[Dict[str, Any]] = []
            for idx, score in scored[: self.top_k]:
                doc = mappings[idx].copy()
                doc["score"] = round(score, 4)
                results.append(doc)

            logger.info(f"✅ SPLADE: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"❌ SPLADE search error: {e}")
            return []
