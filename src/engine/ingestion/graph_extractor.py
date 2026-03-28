import re
from collections import Counter
from typing import List, Dict
from src.core.logger import logger


class GraphExtractor:
    """Trích xuất Keywords & Entities từ chunk transcript mà KHÔNG cần gọi LLM.

    Tối ưu so với bản cũ:
    - Bản cũ: stub mock, trả về hardcoded list
    - Bản mới: Rule-based extraction đủ dùng cho BM25 / metadata filtering:
        1. Capitalized Phrases (Danh từ riêng tiếng Anh)
        2. Số & đơn vị đặc thù (thời gian, tỉ lệ: "3 rounds", "5 seconds")
        3. All-cap abbreviations (AI, RAG, NLP...)
        4. TF-style keyword scoring (top-N từ xuất hiện nhiều nhất, bỏ stopword)
    - Chạy đồng bộ (sync), chi phí CPU cực thấp (<1ms/chunk)
    - Chunk quá ngắn (<150 chars) → bỏ qua để tiết kiệm tài nguyên
    """

    # Stopwords tiếng Anh tối giản (bổ sung thêm nếu cần)
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "this", "that", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
        "should", "may", "might", "it", "its", "you", "your", "we", "our", "they",
        "their", "he", "she", "his", "her", "i", "me", "my", "by", "from", "so",
        "if", "as", "up", "out", "then", "when", "where", "which", "who", "what",
        "how", "all", "each", "any", "one", "two", "not", "no", "more", "also",
        "just", "than", "into", "about", "there", "here", "now", "see", "get",
    }

    def __init__(self, min_chunk_length: int = 150, max_keywords: int = 10):
        self.min_chunk_length = min_chunk_length
        self.max_keywords = max_keywords

    def _extract_capitalized_phrases(self, text: str) -> List[str]:
        """Lấy các cụm danh từ riêng (liên tiếp, viết hoa đầu chữ).
        Ví dụ: 'Bomb Busters', 'Neural Network', 'Boardgame Ninja'
        """
        # Tìm chuỗi từ viết hoa đầu chữ liên tiếp (2+ từ hoặc 1 từ đứng giữa câu)
        pattern = r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*)\b"
        matches = re.findall(pattern, text)
        # Chỉ giữ lại từ 2 ký tự trở lên, không trùng stopwords
        phrases = [m for m in matches if m.lower() not in self.STOPWORDS and len(m) > 3]
        return list(set(phrases))

    def _extract_abbreviations(self, text: str) -> List[str]:
        """Lấy các từ viết tắt (ALL CAPS, 2-6 ký tự).
        Ví dụ: AI, RAG, NLP, API, LLM
        """
        pattern = r"\b([A-Z]{2,6})\b"
        return list(set(re.findall(pattern, text)))

    def _extract_numeric_phrases(self, text: str) -> List[str]:
        """Lấy các cụm số + đơn vị đặc thù.
        Ví dụ: '3 rounds', '5 seconds', '10 players', '8 minutes'
        """
        pattern = r"\b(\d+(?:\.\d+)?\s+(?:rounds?|seconds?|minutes?|hours?|players?|points?|cards?|tokens?|tiles?|dice|sides?))\b"
        return [m.strip() for m in re.findall(pattern, text, re.IGNORECASE)]

    def _extract_top_keywords(self, text: str) -> List[str]:
        """TF-style: Top-N từ xuất hiện nhiều nhất sau khi lọc stopwords.
        Phù hợp làm từ khóa BM25.
        """
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        filtered = [w for w in words if w not in self.STOPWORDS]
        counter = Counter(filtered)
        # Lấy top-N, bỏ những từ chỉ xuất hiện 1 lần (quá hiếm)
        top = [word for word, count in counter.most_common(self.max_keywords * 2) if count >= 2]
        return top[:self.max_keywords]

    def extract_entities(self, text: str) -> List[str]:
        """Tổng hợp tất cả entity từ chunk."""
        entities = []
        entities.extend(self._extract_capitalized_phrases(text))
        entities.extend(self._extract_abbreviations(text))
        entities.extend(self._extract_numeric_phrases(text))
        entities.extend(self._extract_top_keywords(text))

        # Dedup, giữ thứ tự (capitalized phrases ưu tiên trước)
        seen = set()
        result = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                result.append(e)

        return result[:self.max_keywords * 2]  # Cap tổng số entities

    def process_chunk(self, chunk: Dict) -> Dict:
        """Nhét entities vào metadata của chunk. Trả về chunk đã được enrich."""
        content = chunk.get("content", "")

        if len(content) < self.min_chunk_length:
            logger.debug(f"[GraphExtractor] Chunk quá ngắn ({len(content)} chars) → bỏ qua")
            chunk["metadata"]["keywords"] = ""
            return chunk

        entities = self.extract_entities(content)

        if entities:
            chunk["metadata"]["keywords"] = ", ".join(entities)
            logger.debug(f"[GraphExtractor] Trích xuất {len(entities)} entities: {entities[:5]}...")
        else:
            chunk["metadata"]["keywords"] = ""

        return chunk
