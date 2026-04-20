import re
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from src.core.logger import logger


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Tính độ tương đồng Cosine giữa 2 vector."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class SemanticChunker:
    """Chia nhỏ Transcript YouTube thành các đoạn (chunks) dựa trên Ngữ nghĩa (Semantic) + Thời gian (Pause-aware).
    
    Quy trình SOTA 2 giai đoạn:
    1. Atomic Splitting: Gộp các dòng transcript ngắn thành các 'Câu' (Atomic sentence) 
       dựa trên độ trễ giữa người nói (Pause-aware > 1.5s) và dấu câu (., ?, !).
    2. Vector Semantic Grouping: Nhúng các câu thành Vector, tính Cosine Similarity giữa các câu liên tiếp.
       Nhận diện các 'thung lũng ngữ nghĩa' (Semantic valleys) - nơi chuỗi chủ đề bị ngắt trớn, 
       để cắt thành một Chunk hoàn chỉnh.
    """

    def __init__(self, 
                 percentile_threshold: int = 15, 
                 pause_threshold_sec: float = 1.5,
                 min_chars_per_chunk: int = 200,
                 max_chars_per_chunk: int = 2000,
                 **kwargs):
        # Chọn độ dốc ở phần trăm thứ N. (Ví dụ 15: lấy top 15% những cú rớt dốc thảm nhất trong cả video làm điểm ngắt)
        self.percentile_threshold = percentile_threshold 
        self.pause_threshold = pause_threshold_sec
        self.min_chars = min_chars_per_chunk
        self.max_chars = max_chars_per_chunk
        self.llm_client = kwargs.get('llm_client', None)
        
        # Load mô hình nhúng siêu nhẹ để tính toán semantic similarity siêu tốc
        model_name = "all-MiniLM-L6-v2"  
        logger.info(f"Đang tải siêu mô hình nhúng {model_name} cho Semantic Chunking...")
        self.encoder = SentenceTransformer(model_name)

    def _correct_punctuation(self, text: str) -> str:
        """Dùng LLM (nếu có) để phục hồi dấu chấm câu cho Transcript thô."""
        if not self.llm_client:
            return text
            
        try:
            prompt = f"Add appropriate punctuation (periods, commas, question marks) to this text without changing any words or spelling. Reply ONLY with the punctuated text:\n\n{text}"
            corrected = self.llm_client.chat_complete(
                prompt=prompt,
                system="You are a punctuation restoration bot. ONLY return the punctuated text. No extra words.",
                max_tokens=600,
                temperature=0.0,
                model="llama-3.1-8b-instant"
            )
            return corrected.strip() if corrected else text
        except Exception as e:
            logger.warning(f"Lỗi Punctuation Correction: {e}")
            return text

    def _build_atomic_sentences(self, raw_transcript: List[Dict]) -> List[Dict]:
        """Bước 1: Pause-Aware & Punctuation Grouping
        Gộp các dòng transcript rời rạc thành từng 'câu' hoàn chỉnh (Atomic).
        - Nếu có LLM: Sẽ dịch vụ Auto-Punctuation trước khi dò dấu câu.
        """
        atomic_sentences = []
        current_text = []
        start_time = None
        last_end_time = 0.0
        
        # Tiền xử lý (SOTA): Phục hồi dấu câu bằng LLM cho cả cục text dài (Batch 20 dòng)
        # Để tiết kiệm API, ta gộp nội dung lại phục hồi 1 lần rồi đem đè xuống
        # (Để đơn giản cho bản patch này, ta chỉ áp LLM lên text ở runtime nhỏ)
        
        for i, row in enumerate(raw_transcript):
            text = row["text"].strip()
            if not text:
                continue
                
            ts_start = row["start"]
            ts_end = ts_start + row.get("duration", 0)
            
            # Ghi nhận start_time của câu hiện tại
            if start_time is None:
                start_time = ts_start
                
            # Kiểm tra khoảng lặng (pause)
            pause_duration = ts_start - last_end_time
            has_long_pause = (pause_duration > self.pause_threshold) and (i > 0)
            
            # Nếu có khoảng lặng dài -> ÉP ngắt câu ở dòng TRƯỚC (đã add rồi), dòng này là câu mới
            if has_long_pause and current_text:
                joined_text = " ".join(current_text)
                # Auto Punctuation SOTA cho câu hạt nhân 
                if self.llm_client:
                    joined_text = self._correct_punctuation(joined_text)
                    
                atomic_sentences.append({
                    "text": joined_text,
                    "start": start_time,
                    "end": last_end_time
                })
                current_text = [text]
                start_time = ts_start
            else:
                current_text.append(text)
            
            # Kiểm tra dấu câu kết thúc (. ! ?)
            # Nếu Transcript có dấu thì ngắt
            ends_with_punct = re.search(r'[.!?]$', text)
            
            if ends_with_punct:
                joined_text = " ".join(current_text)
                atomic_sentences.append({
                    "text": joined_text,
                    "start": start_time,
                    "end": ts_end
                })
                current_text = []
                start_time = None
                
            last_end_time = ts_end
            
        # Thêm phần còn dư ở cuối video
        if current_text:
            atomic_sentences.append({
                "text": " ".join(current_text),
                "start": start_time,
                "end": last_end_time
            })
            
        return atomic_sentences

    def chunk_document(self, metadata: Dict, raw_transcript: List[Dict]) -> List[Dict]:
        """Bước 2: Vector Semantic Grouping
        Cắt chunk bằng cách dò 'Thung lũng ngữ nghĩa' giữa các Atomic sentences.
        """
        video_id = metadata.get("video_id", "unknown")
        logger.info(f"📐 Bắt đầu SOTA Semantic Chunking: video_id={video_id}")

        if not raw_transcript:
            logger.warning("Transcript rỗng cmnr. Bỏ qua.")
            return []

        # 1. Gộp các dòng rời rạc thành các 'Câu hạt nhân' (Atomic sentences)
        atomic_sentences = self._build_atomic_sentences(raw_transcript)
        if not atomic_sentences:
            return []
            
        # Nếu video quá ngắn, chỉ nói được 1 câu thì đóng chunk luôn
        if len(atomic_sentences) == 1:
            sent = atomic_sentences[0]
            return [{
                "content": sent["text"],
                "metadata": {
                    "video_id": video_id,
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "start_time": round(sent["start"], 2),
                    "end_time": round(sent["end"], 2),
                    "source": metadata.get("video_url", ""),
                    "chunk_index": 0,
                    "char_count": len(sent["text"]),
                    "word_count": len(sent["text"].split()),
                }
            }]

        logger.info(f"   Tìm thấy {len(atomic_sentences)} Atomic Sentences. Đang tiến hành nhúng Vector để dò Topic...")

        # 2. Nhúng Vector cho tất cả các câu
        texts = [s["text"] for s in atomic_sentences]
        embeddings = self.encoder.encode(texts)
        
        # 3. Tính Cosine Similarity giữa Câu N và Câu N+1
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)
            
        # TÍNH NGƯỠNG CẮT ĐỘNG (Dynamic Threshold) THEO SOTA PERCENTILE
        # Tìm ngưỡng điểm Similarity. Ví dụ percentile=15% -> ngưỡng này là mức điểm mà chỉ có 15% 
        # thung lũng sâu nhất mới lọt xuống được. Nó co giãn theo TỪNG VIDEO MỘT!
        dynamic_threshold = np.percentile(similarities, self.percentile_threshold)
        logger.info(f"   [Dynamic Threshold] Tính bằng Percentile thứ {self.percentile_threshold}: Ngưỡng cắt = {dynamic_threshold:.4f}")
            
        # 4. Gom nhóm Câu vào Chunks dựa trên 'Thung lũng ngữ nghĩa'
        chunks = []
        current_chunk_sentences = [atomic_sentences[0]]
        current_char_count = len(atomic_sentences[0]["text"])

        for i, sim_score in enumerate(similarities):
            next_sentence = atomic_sentences[i+1]
            next_char_count = len(next_sentence["text"])
            
            # Điều kiện băm Chunk:
            # - Băm Semantic: Topic score rớt qua thung lũng động (Chuyển chủ đề) + chunk đã dài đủ Min (> 200 chữ)
            # - Băm Bạo lực: Chunk đã quá dài (> 2000 chữ) tránh tràn Limit Context
            is_semantic_break = (sim_score < dynamic_threshold) and (current_char_count >= self.min_chars)
            is_length_break = (current_char_count + next_char_count) > self.max_chars
            
            if is_semantic_break or is_length_break:
                # Chốt sổ chunk hiện tại
                chunk_text = " ".join([s["text"] for s in current_chunk_sentences])
                chunks.append({
                    "content": chunk_text,
                    "start_time": current_chunk_sentences[0]["start"],
                    "end_time": current_chunk_sentences[-1]["end"],
                })
                # Bắt đầu chunk mới
                current_chunk_sentences = [next_sentence]
                current_char_count = next_char_count
            else:
                # Nhồi thêm câu tiếp theo vào chunk đang chứa
                current_chunk_sentences.append(next_sentence)
                current_char_count += next_char_count + 1 # Cộng 1 khoảng trắng
                
        # Chốt sổ đoạn văn cuối cùng
        if current_chunk_sentences:
            chunk_text = " ".join([s["text"] for s in current_chunk_sentences])
            chunks.append({
                "content": chunk_text,
                "start_time": current_chunk_sentences[0]["start"],
                "end_time": current_chunk_sentences[-1]["end"],
            })
            
        # 5. Pack vào Class Metadata chuẩn chỉnh trả về
        processed_chunks = []
        for i, c in enumerate(chunks):
            processed_chunks.append({
                "content": c["content"],
                "metadata": {
                    "video_id": video_id,
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "start_time": round(c["start_time"], 2),
                    "end_time": round(c["end_time"], 2),
                    "source": metadata.get("video_url", ""),
                    "chunk_index": i,
                    "char_count": len(c["content"]),
                    "word_count": len(c["content"].split()),
                }
            })

        logger.info(
            f"✅ SOTA Chunking Hoàn Tất: {len(raw_transcript)} sub lines -> {len(atomic_sentences)} sentences -> chốt thành {len(processed_chunks)} siêu Chunks!"
        )
        return processed_chunks
