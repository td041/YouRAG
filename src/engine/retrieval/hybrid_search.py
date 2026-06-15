from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from src.engine.retrieval.dense_search import DenseRetriever
from src.engine.retrieval.sparse_search import SparseRetriever
from src.core.logger import logger

class HybridRetriever:
    """Hybrid Search kết hợp hai thế lực:
    - Dense Search (Vector, ý nghĩa từ BAAI/bge-m3)
    - Sparse Search (BM25, từ khóa chính xác)
    Sử dụng Reciprocal Rank Fusion (RRF) để kết hợp kết quả.
    """

    def __init__(self, top_k: int = 5, rrf_k: int = 60, alpha: float = 0.5):
        """
        Args:
            top_k: Số lượng document trả về cuối cùng
            rrf_k: Hằng số tinh chỉnh RRF (tránh việc rank 1 được ưu tiên quá lố)
            alpha: Trọng số giữa Dense và Sparse. 0.5 = cân đối. (0 = BM25, 1 = Dense)
        """
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.alpha = alpha
        
        # Singleton retrievers — BM25 index cached across requests
        self.dense = DenseRetriever(top_k=top_k * 2)
        self.sparse = SparseRetriever(top_k=top_k * 2)

    def search(self, query: str, collection_name: str, filters: dict = None) -> List[Dict[str, Any]]:
        """Tìm kiếm Hybrid sử dụng Reciprocal Rank Fusion (RRF)."""
        logger.info(f"✨ Hybrid Search (RRF): '{query}' -> [{collection_name}]")
        
        # 1. Dense + Sparse chạy song song — tiết kiệm ~300ms mỗi query
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self.dense.search, query, collection_name, filters)
            future_sparse = executor.submit(self.sparse.search, query, collection_name)
            dense_results = future_dense.result()
            sparse_results = future_sparse.result()
        
        if not dense_results and not sparse_results:
            return []

        # 2. Thuật toán RRF: Tính điểm fusion cho từng document ID
        rrf_scores = {}
        doc_contents = {}

        # 2A. RRF cho Dense (Vector)
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_contents[doc_id] = doc
            
            # Tính điểm RRF Dense: (1 / (rrf_k + rank)) * tỷ trọng Dense (alpha)
            rrf_scores[doc_id] += self.alpha * (1.0 / (self.rrf_k + rank))

        # 2B. RRF cho Sparse (BM25)
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_contents[doc_id] = doc
                
            # Tính điểm RRF Sparse: (1 / (rrf_k + rank)) * tỷ trọng Sparse (1 - alpha)
            rrf_scores[doc_id] += (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank))

        # 3. Sắp xếp kết quả sau khi fusion
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 4. Trả về top K cuối cùng
        final_results = []
        for doc_id, final_score in sorted_rrf[:self.top_k]:
            doc = doc_contents[doc_id].copy()
            doc["hybrid_score"] = round(final_score, 5)
            # Dọn đi các rác của từng thuật toán để UI sạch
            doc.pop("score", None)
            doc.pop("distance", None)
            final_results.append(doc)

        logger.info(f"✅ Hybrid Search thành công (top {len(final_results)} results)")
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
                collection_name = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Search failed for {collection_name}: {e}")

        # Sort tất cả chunks từ mọi video theo hybrid_score, lấy top k*2
        all_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return all_results[: self.top_k * 2]
