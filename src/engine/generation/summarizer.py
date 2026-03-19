from typing import Optional
from src.engine.generation.llm_client import LLMClient
from src.core.database import db_instance
from src.core.logger import logger
from src.core.utils import format_timestamp

class VideoSummarizer:
    """Module Tóm tắt Toàn bộ nội dung Video.
    Kéo các chunk từ Qdrant, sắp xếp lại theo thời gian và nhờ LLM đọc/tóm tắt lại.
    """

    def __init__(self):
        # Không truyền model cứng vào đây để LLMClient tự lấy model mặc định của Provider (VD: Qwen của Ollama)
        self.llm = LLMClient()
        self.db = db_instance

    def summarize(self, collection_name: str) -> Optional[str]:
        """Tóm tắt một video đã được Ingest trong hệ thống."""
        logger.info(f"📝 Bắt đầu tóm tắt video từ collection: '{collection_name}'...")
        
        try:
            records = []
            offset = None
            while True:
                result, next_offset = db_instance.client.scroll(
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
                logger.error("❌ Collection rỗng hoặc không tồn tại.")
                return None
            
            # Gỡ các chunk ra và sắp xếp lại theo thời gian thực của Video
            chunks = []
            for rec in records:
                payload = rec.payload or {}
                chunks.append({
                    "content": payload.get("text", ""),
                    "index": payload.get("chunk_index", 0),
                    "start_time": payload.get("start_time", 0.0)
                })
                
            # Sort theo thứ tự gốc từ đầu đến cuối video
            chunks.sort(key=lambda x: x["index"])
            
            # Khống chế số lượng Chunks tối đa để lọt qua khe cửa hẹp 6000 TPM của Groq Free
            # Trung bình 1 chunk khoảng 800 ký tự -> 12 chunks (10000 ký tự) là an toàn nhất
            MAX_CHUNKS = 12
            if len(chunks) > MAX_CHUNKS:
                logger.warning(f"Video quá dài ({len(chunks)} chunks). Đang áp dụng Trích mẫu Tỉ Lệ Đều (Uniform Sampling)...")
                step = len(chunks) / MAX_CHUNKS
                # Trích 12 chunks cách đều nhau từ đầu đến cuối video
                sampled_chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS)]
                chunks = sampled_chunks
            
            # Ghép lại thành 1 văn bản dài kèm mốc thời gian
            full_transcript = []
            for c in chunks:
                # Bỏ qua phần ngữ cảnh (context) mà Contextual Enrichment chèn vào 
                # (để tránh lặp từ khi LLM đọc tóm tắt).
                raw_text = c["content"].split("\n\n")[-1] if "\n\n" in c["content"] else c["content"]
                
                # Format: [mm:ss] Nội dung đoạn nói
                ts = format_timestamp(c['start_time'])
                full_transcript.append(f"[{ts}] {raw_text}")
                
            transcript_text = "\n".join(full_transcript)

            # Prompt yêu cầu AI tóm tắt
            prompt = f"""
Dưới đây là phần kịch bản (transcript) của một video, với các mốc thời gian (tính bằng giây) đi kèm ở đầu mỗi đoạn:

<transcript>
{transcript_text}
</transcript>

Nhiệm vụ của bạn là đóng vai một chuyên gia phân tích video. Hãy viết một bản TÓM TẮT CỰC KỲ CHI TIẾT VÀ CHUYÊN SÂU về video này bằng TIẾNG VIỆT tự nhiên.
Yêu cầu định dạng nghiêm ngặt (Markdown):
1. **Giới thiệu Tổng quan**: Viết 1 đoạn văn (3-5 câu) tóm lược toàn cảnh video này nói về chủ đề / sự kiện / bộ môn gì? Mục tiêu của video là gì?
2. **Nội dung Chi tiết (Diễn biến)**:
   - Chia thành nhiều gạch đầu dòng (bullet points).
   - MỖI gạch đầu dòng phải mô tả RẤT DÀI VÀ RÕ RÀNG về một phân đoạn kiến thức, mẹo, luật chơi, hoặc sự kiện quan trọng trong video. Đừng chỉ tóm tắt 1 câu ngắn ngủi.
3. **Mốc thời gian (BẮT BUỘC)**: Cuối mỗi gạch đầu dòng, BẮT BUỘC chèn mốc thời gian chính xác theo định dạng ngoặc vuông `[mm:ss]` hoặc `[Xs]` (Ví dụ: `[1:30]` hoặc `[45.2s]`). Điều này để giúp khán giả bấm vào xem đúng đoạn đó.
4. **Kết luận / Lời khuyên**: 1 đoạn tổng kết ngắn.

Hãy phân tích thật sâu sát, không bỏ sót các mẹo vặt hay điểm nhấn quan trọng nào của video!
"""
            # Groq Free Tier giới hạn 6000 Token 1 phút dùng chung (Cả Prompt vào + Chữ ra)
            # Ta giảm max_tokens về 1500 để nhường phần lớn token cho việc đọc Video dài
            summary = self.llm.chat_complete(
                prompt=prompt,
                system="Bạn là một chuyên gia phân tích dữ liệu video. Bạn cực kỳ cẩn thận, không được bỏ sót chi tiết và phải giải thích cặn kẽ mọi thứ bằng tiếng Việt.",
                max_tokens=1500, # Giảm xuống 1500 tránh crash Groq
                temperature=0.3  # Giữ ở mức logic nhưng văn phong tự nhiên
            )
            
            logger.info("✅ Quá trình tóm tắt hoàn tất!")
            return summary

        except Exception as e:
            logger.error(f"❌ Lỗi khi tóm tắt video: {e}")
            return None

    def summarize_stream(self, collection_name: str):
        """Tóm tắt video theo kiểu Streaming (nhả chữ dần dần)."""
        logger.info(f"📝 Bắt đầu tóm tắt STREAM từ collection: '{collection_name}'...")
        
        try:
            records = []
            offset = None
            while True:
                result, next_offset = db_instance.client.scroll(
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
                yield "❌ Lỗi: Video chưa được xử lý hoặc không tồn tại."
                return

            chunks = []
            for rec in records:
                payload = rec.payload or {}
                chunks.append({
                    "content": payload.get("text", ""),
                    "index": payload.get("chunk_index", 0),
                    "start_time": payload.get("start_time", 0.0)
                })
            chunks.sort(key=lambda x: x["index"])

            # Uniform Sampling
            MAX_CHUNKS = 12
            if len(chunks) > MAX_CHUNKS:
                step = len(chunks) / MAX_CHUNKS
                chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS)]
            
            full_transcript = []
            for c in chunks:
                raw_text = c["content"].split("\n\n")[-1] if "\n\n" in c["content"] else c["content"]
                ts = format_timestamp(c['start_time'])
                full_transcript.append(f"[{ts}] {raw_text}")
                
            transcript_text = "\n".join(full_transcript)

            prompt = f"""
Dưới đây là phần kịch bản trích mẫu của một video:
<transcript>
{transcript_text}
</transcript>

Hãy viết một bản TÓM TẮT CHI TIẾT VÀ CHUYÊN SÂU bằng TIẾNG VIỆT.
Yêu cầu:
1. Giới thiệu tổng quan.
2. Nội dung chi tiết từng phần kèm mốc thời gian [mm:ss] ở cuối mỗi ý.
3. Kết luận.
Hãy viết văn phong chuyên nghiệp, dễ hiểu.
"""
            # Gọi streaming
            for chunk in self.llm.chat_complete_stream(
                prompt=prompt,
                system="Bạn là một chuyên gia phân tích dữ liệu video. Hãy tóm tắt thật hay và đầy đủ.",
                max_tokens=2000,
                temperature=0.3
            ):
                yield chunk

        except Exception as e:
            logger.error(f"❌ Lỗi summarize stream: {e}")
            yield f"❌ Lỗi hệ thống: {str(e)}"
