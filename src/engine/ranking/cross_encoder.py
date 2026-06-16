from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import torch
from src.core.config import settings
from src.core.logger import logger

class CrossEncoderReranker:
    """Class sử dụng mô hình Cross-Encoder để đánh giá (rerank) lại độ phù hợp của 
    các chunk văn bản so với câu hỏi (query).
    
    Khác với Bi-Encoder (so khớp khoảng cách vector), Cross-Encoder tính toán sự 
    kết nối sâu sắc bằng cách cho cả Query và Document vào Model cùng một lúc, 
    cho ra điểm đánh giá chính xác tuyệt đối (nhưng chậm hơn và đòi hỏi tài nguyên).
    """

    def __init__(self):
        # BGE-reranker-v2-m3: SOTA multilingual reranker (~568M), tiếng Việt tốt hơn mmarco.
        # Vừa GPU 6GB cùng bge-m3. Fallback về chunks gốc nếu load lỗi.
        self.model_name = getattr(settings, "CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

        if self.device == "cpu":
            logger.warning(
                "⚠️ Reranker chạy trên CPU — model 568M sẽ chậm hơn GPU. "
                "Thêm GPU hoặc set CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 để nhanh hơn."
            )

        logger.info(f"⏳ Đang tải Cross-Encoder Model: {self.model_name} (Device: {self.device})...")
        try:
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("✅ Load Cross-Encoder Model thành công!")
        except Exception as e:
            logger.error(f"❌ Lỗi tải Cross-Encoder: {e}")
            self._model = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Nhận một list các Dict văn bản (từ Hybrid hoặc DenseSearch), chấm điểm 
        lại bằng Cross-Encoder, sau đó lọc ra Top K tối ưu nhất.
        """
        if not chunks:
            return []
            
        if self._model is None:
            logger.warning("⚠️ Cross-Encoder chưa được tải. Bỏ qua bước Rerank và trả về kết quả gốc.")
            return chunks[:top_k]

        logger.info(f"⚖️ Bắt đầu Reranking {len(chunks)} chunks cho truy vấn: '{query}'")
        try:
            # Tạo list các cặp (Query, Chunk_Content) chuyên biệt cho bge-reranker
            pairs = [[query, chunk["content"]] for chunk in chunks]
            
            # Tính điểm (logit score, không bị ép về 0-1)
            scores = self._model.predict(pairs)
            
            # Gán điểm vào từng chunk
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
                
            # Sắp xếp lại dựa theo độ cao của điểm rerank
            reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            
            # Lấy top K tốt nhất
            final_results = reranked_chunks[:top_k]
            logger.info(f"✅ Reranking xong! Cắt giảm còn lại {len(final_results)} top chunks.")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Lỗi Rerank: {e}")
            return chunks[:top_k] # Nếu lỗi, fallback về mặc định
