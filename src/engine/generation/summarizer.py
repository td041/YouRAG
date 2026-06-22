from typing import Optional
from src.engine.generation.llm_client import LLMClient
from src.core.database import db_instance
from src.core.logger import logger
from src.core.utils import format_timestamp
from src.core.config import settings

_SUMMARY_KEY_PREFIX = "summary:"


def _get_redis():
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        r.ping()
        return r
    except Exception:
        return None


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
        cache_key = f"{_SUMMARY_KEY_PREFIX}{collection_name}"
        r = _get_redis()
        if r:
            cached = r.get(cache_key)
            if cached:
                logger.info(f"⚡ Summary cache hit for [{collection_name}]")
                return cached

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
            
            if not chunks:
                logger.error("❌ Collection không có chunks hợp lệ.")
                return None

            MAX_CHUNKS = settings.SUMMARIZER_MAX_CHUNKS
            if len(chunks) > MAX_CHUNKS:
                logger.warning(f"Video quá dài ({len(chunks)} chunks). Đang áp dụng Trích mẫu Tỉ Lệ Đều...")
                step = len(chunks) / MAX_CHUNKS
                chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS)]
            
            # Ghép lại thành 1 văn bản dài kèm mốc thời gian
            # Truncate mỗi chunk để tránh vượt TPM limit của Groq free tier (6000 TPM)
            MAX_CHARS_PER_CHUNK = 300
            full_transcript = []
            for c in chunks:
                # Bỏ qua phần ngữ cảnh (context) mà Contextual Enrichment chèn vào
                raw_text = c["content"].split("\n\n")[-1] if "\n\n" in c["content"] else c["content"]
                raw_text = raw_text[:MAX_CHARS_PER_CHUNK]

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
            summary = self.llm.chat_complete(
                prompt=prompt,
                system="Bạn là một chuyên gia phân tích dữ liệu video. Hãy tóm tắt cô đọng bằng tiếng Việt.",
                max_tokens=2500,
                temperature=0.3
            )
            
            logger.info("✅ Quá trình tóm tắt hoàn tất!")
            if summary and r:
                r.set(cache_key, summary)
                logger.info(f"💾 Summary cached to Redis [{collection_name}]")
            return summary

        except Exception as e:
            logger.error(f"❌ Lỗi khi tóm tắt video: {e}")
            return None

    def summarize_stream(self, collection_name: str):
        """Tóm tắt video theo kiểu Streaming (nhả chữ dần dần)."""
        cache_key = f"{_SUMMARY_KEY_PREFIX}{collection_name}"
        r = _get_redis()
        if r:
            cached = r.get(cache_key)
            if cached:
                logger.info(f"⚡ Summary stream cache hit for [{collection_name}]")
                yield cached
                return

        logger.info(f"📝 Bắt đầu tóm tắt STREAM từ collection: '{collection_name}'...")
        accumulated: list[str] = []

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

            # Uniform Sampling — keep under Groq free-tier TPM (6000 tokens/min)
            MAX_CHUNKS = settings.SUMMARIZER_MAX_CHUNKS
            if len(chunks) > MAX_CHUNKS:
                step = len(chunks) / MAX_CHUNKS
                chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS)]

            MAX_CHARS_PER_CHUNK = 300
            full_transcript = []
            for c in chunks:
                raw_text = c["content"].split("\n\n")[-1] if "\n\n" in c["content"] else c["content"]
                raw_text = raw_text[:MAX_CHARS_PER_CHUNK]
                ts = format_timestamp(c['start_time'])
                full_transcript.append(f"[{ts}] {raw_text}")
                
            transcript_text = "\n".join(full_transcript)

            prompt = f"""
Dưới đây là phần kịch bản trích mẫu của một video, với mốc thời gian đi kèm:

<transcript>
{transcript_text}
</transcript>

Hãy viết một bản TÓM TẮT CHI TIẾT VÀ CHUYÊN SÂU bằng TIẾNG VIỆT tự nhiên, dùng định dạng Markdown.
Yêu cầu:
1. **Giới thiệu Tổng quan** — 1 đoạn văn (3-5 câu) nói video nói về chủ đề gì, mục tiêu là gì.
2. **Nội dung Chi tiết (Diễn biến)** — gạch đầu dòng cho từng phân đoạn. Dùng **in đậm** cho khái niệm, thuật ngữ, điểm mấu chốt quan trọng. Cuối mỗi ý thêm mốc thời gian `[mm:ss]`.
3. **Kết luận / Lời khuyên** — tổng kết ngắn gọn.
Viết văn phong chuyên nghiệp, không bỏ sót điểm nhấn quan trọng.
"""
            for chunk in self.llm.chat_complete_stream(
                prompt=prompt,
                system="Bạn là một chuyên gia phân tích dữ liệu video. Hãy tóm tắt thật hay và đầy đủ.",
                max_tokens=2500,
                temperature=0.3
            ):
                accumulated.append(chunk)
                yield chunk

            if accumulated and r:
                full = "".join(accumulated)
                r.set(cache_key, full)
                logger.info(f"💾 Summary stream cached to Redis [{collection_name}]")

        except Exception as e:
            logger.error(f"❌ Lỗi summarize stream: {e}")
            yield f"❌ Lỗi hệ thống: {str(e)}"
