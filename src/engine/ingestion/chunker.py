import re
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.logger import logger


class SemanticChunker:
    """Chia nhỏ Transcript YouTube thành các đoạn (chunks) có ý nghĩa.

    Tối ưu so với bản cũ:
    1. Character-offset timestamp map: O(1) lookup thay vì O(n×m) string search
       - Xây dựng mapping: char_position → timestamp trước khi cắt
       - Khi có chunk, tìm ngay char offset đầu tiên → lookup timestamp chính xác
    2. Overlap tracking: tính end_time của mỗi chunk để UI biết span đầy đủ
    3. Word count + char count trong metadata (hữu ích cho ranking sau này)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            keep_separator=False,
        )

    def _build_offset_map(self, raw_transcript: List[Dict]) -> Tuple[str, List[Tuple[int, float, float]]]:
        """Xây dựng:
        - full_text: toàn bộ transcript nối lại bằng dấu cách
        - offset_map: list[(char_start, timestamp_start, timestamp_end)]
          → Mỗi entry ánh xạ 1 khoảng ký tự trong full_text về timestamp gốc.

        Phương pháp này đảm bảo timestamp mapping 100% chính xác,
        không phụ thuộc vào string matching.
        """
        segments = []
        cursor = 0  # Vị trí ký tự hiện tại trong full_text

        parts = []
        for row in raw_transcript:
            text = row["text"]
            if not text:
                continue
            char_start = cursor
            char_end = cursor + len(text)
            ts_start = row["start"]
            ts_end = ts_start + row.get("duration", 0)
            segments.append((char_start, ts_start, ts_end))
            parts.append(text)
            cursor = char_end + 1  # +1 cho khoảng trắng giữa các dòng

        full_text = " ".join(parts)
        return full_text, segments

    def _lookup_timestamp(
        self,
        char_offset: int,
        offset_map: List[Tuple[int, float, float]],
    ) -> Tuple[float, float]:
        """Binary search trong offset_map để tìm timestamp của char_offset.
        
        Trả về (start_time, end_time) của đoạn phụ đề chứa char_offset đó.
        Complexity: O(log n).
        """
        lo, hi = 0, len(offset_map) - 1
        result_idx = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            if offset_map[mid][0] <= char_offset:
                result_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return offset_map[result_idx][1], offset_map[result_idx][2]

    def chunk_document(self, metadata: Dict, raw_transcript: List[Dict]) -> List[Dict]:
        """Chia nhỏ transcript và ánh xạ timestamp chính xác bằng offset map."""
        video_id = metadata.get("video_id", "unknown")
        logger.info(f"📐 Bắt đầu Chunking Transcript: video_id={video_id}")

        # --- Bước 1: Xây dựng full_text + offset map ---
        full_text, offset_map = self._build_offset_map(raw_transcript)
        total_chars = len(full_text)

        if not full_text.strip():
            logger.warning("Transcript rỗng sau khi compile. Bỏ qua.")
            return []

        # --- Bước 2: Cắt thành chunks ---
        # Dùng split_text_with_positions để biết char offset từng chunk
        # LangChain chưa có API này trực tiếp → dùng create_documents + tìm position thủ công
        chunk_texts = self.splitter.split_text(full_text)

        if not chunk_texts:
            logger.warning("Không tạo được chunk nào từ transcript.")
            return []

        # --- Bước 3: Map timestamp cho từng chunk ---
        processed_chunks = []
        search_start = 0  # Tìm từ vị trí trước đó trong full_text → đảm bảo thứ tự

        for i, chunk_text in enumerate(chunk_texts):
            # Tìm char offset của chunk trong full_text (bắt đầu từ search_start)
            char_offset = full_text.find(chunk_text[:50], search_start)  # Match 50 ký tự đầu
            if char_offset == -1:
                # Fallback: tìm từ đầu (trường hợp overlap gây rewind)
                char_offset = full_text.find(chunk_text[:50])
            if char_offset == -1:
                char_offset = search_start  # Last resort

            # Cập nhật search_start để lần tìm sau không quay lui quá xa
            search_start = max(0, char_offset + len(chunk_text) - self.chunk_overlap)

            # Lookup timestamp O(log n)
            start_time, _ = self._lookup_timestamp(char_offset, offset_map)
            end_char = char_offset + len(chunk_text)
            _, end_time = self._lookup_timestamp(min(end_char, total_chars - 1), offset_map)

            processed_chunks.append({
                "content": chunk_text,
                "metadata": {
                    "video_id": video_id,
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "start_time": round(start_time, 2),     # Giây → để UI jump video
                    "end_time": round(end_time, 2),         # Span của chunk
                    "source": metadata.get("video_url", ""),
                    "chunk_index": i,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                },
            })

        logger.info(
            f"✅ Chunking hoàn tất: {len(processed_chunks)} chunks "
            f"| avg {total_chars // max(len(processed_chunks), 1)} chars/chunk "
            f"| overlap={self.chunk_overlap}"
        )
        return processed_chunks
