from typing import List, Dict, Any
from src.core.database import db_instance
from src.core.logger import logger

class DenseRetriever:
    """Tìm kiếm Vector Similarity dựa trên HNSW Index của Qdrant.
    Hỗ trợ tìm kiếm xuyên suốt một Collection được chỉ định.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.db = db_instance

    def search(self, query: str, collection_name: str, filters: dict = None) -> List[Dict[str, Any]]:
        """Nhúng câu hỏi (query) và tìm kiếm Vector.

        Args:
            query: Câu hỏi của người dùng
            collection_name: Tên của Collection (thường là tên video đã normalize)
            filters: Metadata filter (vd: lọc theo time_start, author...)

        Returns:
            List các Dict chứa nội dung chunk, metadata và độ tương đồng.
        """
        logger.info(f"🔍 Dense Search Qdrant: '{query}' -> [{collection_name}]")
        try:
            # Nhúng vector câu hỏi
            query_vector = self.db.embedding_model.encode([query])[0]
            
            # Gửi truy vấn Vector sang Qdrant Server
            results = self.db.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=self.top_k
            )

            # Phân tách và trả về dữ liệu chuẩn
            formatted_chunks = []
            
            if not results:
                logger.warning(f"Không tìm thấy kết quả nào cho query: {query}")
                return formatted_chunks
                
            for res in results:
                payload = res.payload or {}
                content = payload.pop("text", "")
                
                formatted_chunks.append({
                    "id": res.id,
                    "content": content,
                    "metadata": payload,
                    # Qdrant dùng Cosine Similarity, score càng to (gần 1.0) càng giống nhau
                    "distance": round(res.score, 4),
                })
            
            return formatted_chunks

        except Exception as e:
            logger.error(f"❌ Lỗi khi Qdrant Dense Search '{query}': {e}")
            return []
