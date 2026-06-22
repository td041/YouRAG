from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import re

from src.engine.retrieval.dense_search import DenseRetriever
from src.engine.retrieval.splade_search import SpladeRetriever
from src.engine.retrieval.query_expander import QueryExpander
from src.core.logger import logger


class HybridRetriever:
    """Hybrid Search kết hợp Dense (bge-m3) + SPLADE sparse + Query Expansion.
    Sử dụng Reciprocal Rank Fusion (RRF) để kết hợp kết quả.
    """

    def __init__(self, top_k: int = 7, rrf_k: int = 60, alpha: float = 0.6):
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.dense = DenseRetriever(top_k=top_k * 2)
        self.sparse = SpladeRetriever(top_k=top_k * 2)
        self._expander = QueryExpander(n=2)
        self._db = None

    @staticmethod
    def _detect_lang(text: str) -> str:
        vi_count = len(re.findall(
            r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',
            text.lower(),
        ))
        return "vi" if vi_count / max(len(text), 1) > 0.03 else "en"

    def _hyde_vector(self, query: str) -> Optional[List[float]]:
        """HyDE: generate hypothetical answer → embed → dùng làm dense query vector.
        Dùng 8b-instant để nhanh. Trả về None nếu fail.
        """
        try:
            from src.engine.generation.llm_client import LLMClient
            from src.core.config import settings
            llm = LLMClient(model=settings.LLM_CONTEXTUAL_MODEL)
            if self._detect_lang(query) == "en":
                prompt = f"Write a short paragraph (2-3 sentences) answering this question:\n{query}"
                system = "Answer directly and concisely."
            else:
                prompt = f"Viết một đoạn văn ngắn (2-3 câu) trả lời câu hỏi sau:\n{query}"
                system = "Trả lời trực tiếp, súc tích."
            hypo = llm.chat_complete(prompt=prompt, system=system, max_tokens=150, temperature=0.3)
            if self._db is None:
                from src.core.database import db_instance
                self._db = db_instance
            vector = self._db.embedding_model.encode([hypo])[0].tolist()
            logger.info(f"[HyDE] hypothetical doc generated ({len(hypo)} chars)")
            return vector
        except Exception as e:
            logger.warning(f"[HyDE] Failed, skipping: {e}")
            return None

    def search(self, query: str, collection_name: str, filters: dict = None) -> List[Dict[str, Any]]:
        """Hybrid Dense+SPLADE với Query Expansion và HyDE (tuỳ độ dài query).
        Tất cả chạy song song trong ThreadPoolExecutor.
        """
        logger.info(f"✨ Hybrid Search (RRF): '{query[:60]}' -> [{collection_name}]")

        _use_hyde = len(query.split()) >= 10
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_dense_base = executor.submit(self.dense.search, query, collection_name, filters)
            future_sparse_base = executor.submit(self.sparse.search, query, collection_name)
            future_expand = executor.submit(self._expander.expand, query)
            future_hyde = executor.submit(self._hyde_vector, query) if _use_hyde else None

            expanded = future_expand.result()
            logger.info(f"   Query expansion: {1 + len(expanded)} queries total")

            futures_dense_extra = [
                executor.submit(self.dense.search, q, collection_name, filters) for q in expanded
            ]
            futures_sparse_extra = [
                executor.submit(self.sparse.search, q, collection_name) for q in expanded
            ]

            dense_results_list = [future_dense_base.result()] + [f.result() for f in futures_dense_extra]
            sparse_results_list = [future_sparse_base.result()] + [f.result() for f in futures_sparse_extra]
            hyde_vec = future_hyde.result() if future_hyde else None

        if hyde_vec:
            hyde_results = self.dense.search_by_vector(hyde_vec, collection_name)
            dense_results_list.append(hyde_results)
            logger.info(f"   HyDE: {len(hyde_results)} results added")

        if not any(dense_results_list) and not any(sparse_results_list):
            return []

        rrf_scores: Dict[Any, float] = {}
        doc_contents: Dict[Any, Dict] = {}

        for dense_results in dense_results_list:
            for rank, doc in enumerate(dense_results, 1):
                doc_id = doc["id"]
                rrf_scores.setdefault(doc_id, 0.0)
                doc_contents[doc_id] = doc
                rrf_scores[doc_id] += self.alpha * (1.0 / (self.rrf_k + rank))

        for sparse_results in sparse_results_list:
            for rank, doc in enumerate(sparse_results, 1):
                doc_id = doc["id"]
                rrf_scores.setdefault(doc_id, 0.0)
                doc_contents[doc_id] = doc
                rrf_scores[doc_id] += (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank))

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = []
        for doc_id, final_score in sorted_rrf[: self.top_k]:
            doc = doc_contents[doc_id].copy()
            doc["hybrid_score"] = round(final_score, 5)
            doc.pop("score", None)
            doc.pop("distance", None)
            final_results.append(doc)

        logger.info(f"✅ Hybrid Search: top {len(final_results)} results")
        return final_results

    def search_multi(self, query: str, collection_names: List[str]) -> List[Dict[str, Any]]:
        """Search nhiều collections song song, merge và sort theo hybrid_score."""
        if len(collection_names) == 1:
            return self.search(query, collection_names[0])

        logger.info(f"🔍 Multi-collection search: {len(collection_names)} videos")
        all_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(len(collection_names), 4)) as executor:
            futures = {executor.submit(self.search, query, name): name for name in collection_names}
            for future in as_completed(futures):
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    logger.error(f"Search failed for {futures[future]}: {e}")

        all_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return all_results[: self.top_k]
