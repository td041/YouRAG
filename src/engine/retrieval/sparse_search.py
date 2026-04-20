from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.core.database import db_instance
from src.core.logger import logger

class SparseRetriever:
    """Tìm kiếm bằng thuật toán từ khóa BM25 (Sparse Retrieval).
    Qdrant chưa hỗ trợ BM25 native triệt để trên Local, nên ta phải load toàn bộ texts lên BM25Okapi in-memory.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.db = db_instance
        # Cache BM25 object mỗi Collection để đỡ tốn công build lại nếu query liên tục
        self._bm25_cache: Dict[str, BM25Okapi] = {}
        self._doc_mappings: Dict[str, List[Dict]] = {} # Lưu id và content gốc

    def _build_or_get_bm25(self, collection_name: str) -> BM25Okapi:
        """Kéo toàn bộ document từ Qdrant trong collection và build chỉ mục BM25."""
        
        # Nếu đã build rồi và cache còn thì lấy luôn
        if collection_name in self._bm25_cache:
            return self._bm25_cache[collection_name]
            
        logger.info(f"⚙️ Building BM25 index cho Collection Qdrant: '{collection_name}'...")
        try:
            records = []
            offset = None
            while True:
                result, next_offset = self.db.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                records.extend(result)
                if next_offset is None:
                    break
                offset = next_offset

            if not records:
                raise ValueError("Collection rỗng!")
                
            docs = []
            ids = []
            metas = []
            for rec in records:
                payload = rec.payload or {}
                docs.append(payload.pop("text", ""))
                ids.append(rec.id)
                metas.append(payload)
            
            if not docs:
                raise ValueError("Collection rỗng!")
                
            # BM25 yêu cầu input là mảng các từ (tokens)
            tokenized_corpus = [doc.lower().split() for doc in docs]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Lưu lại vào cache
            self._bm25_cache[collection_name] = bm25
            
            docs_mapping = []
            for i in range(len(ids)):
                docs_mapping.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i]
                })
            self._doc_mappings[collection_name] = docs_mapping
            
            logger.info("✅ BM25 Index build hoàn tất!")
            return bm25
            
        except Exception as e:
            logger.error(f"❌ Lỗi build BM25 cho {collection_name}: {e}")
            raise

    def search(self, query: str, collection_name: str) -> List[Dict[str, Any]]:
        """Tìm BM25.
        Tra về list dict giống hệt DenseRetriever nhưng score theo chuẩn BM25.
        Lưu ý BM25: Điểm càng CAO càng tốt (ngược với Qdrant distance).
        """
        logger.info(f"🔍 Sparse Search (BM25): '{query}' -> [{collection_name}]")
        try:
            bm25 = self._build_or_get_bm25(collection_name)
            docs_mapping = self._doc_mappings[collection_name]
            
            # Tokenize câu hỏi để BM25 hiểu
            tokenized_query = query.lower().split()
            
            # Nhận mảng điểm số của tất cả các documents
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Sắp xếp lấy Top K documents có điểm cao nhất
            top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:self.top_k]
            
            results = []
            for idx in top_n_indices:
                score = doc_scores[idx]
                # Bỏ qua những doc không dính tí nào tới keyword (score = 0)
                if score <= 0:
                    continue
                    
                doc_info = docs_mapping[idx]
                results.append({
                    "id": doc_info["id"],
                    "content": doc_info["content"],
                    "metadata": doc_info["metadata"],
                    "score": round(score, 4)  # SCORE CAO = GIỐNG NHAU (ngược lại với dense distance)
                })
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi BM25 Search '{query}': {e}")
            return []
