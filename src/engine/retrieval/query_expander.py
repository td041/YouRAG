"""
Query Expander — sinh thêm N câu hỏi thay thế để tăng retrieval recall.

Dùng LLM (llama-3.1-8b-instant) để paraphrase query gốc với vocabulary khác.
Kết quả từ tất cả queries được merge qua RRF trong HybridRetriever.
"""

import json
from typing import List

from src.core.config import settings
from src.core.logger import logger


class QueryExpander:
    """Sinh N câu hỏi thay thế từ query gốc để tăng recall.

    Dùng model nhỏ (8b-instant) để nhanh và tiết kiệm quota.
    Nếu LLM fail, trả về [] — HybridRetriever vẫn dùng query gốc bình thường.
    """

    _PROMPT = (
        'Cho câu hỏi: "{query}"\n\n'
        "Tạo {n} câu hỏi thay thế với cách diễn đạt KHÁC nhau nhưng cùng ý nghĩa. "
        "Mục đích là tìm kiếm tài liệu hiệu quả hơn bằng vocabulary đa dạng.\n\n"
        "Trả về JSON array (không markdown, không giải thích):\n"
        '["câu hỏi 1", "câu hỏi 2"]'
    )

    def __init__(self, n: int = 2):
        self.n = n
        self._llm = None  # lazy init

    def _get_llm(self):
        if self._llm is None:
            from src.engine.generation.llm_client import LLMClient
            self._llm = LLMClient(model=settings.LLM_CONTEXTUAL_MODEL)
        return self._llm

    def expand(self, query: str) -> List[str]:
        """Trả về list các query thay thế (không bao gồm query gốc).
        Trả về [] nếu fail để caller vẫn dùng được query gốc.
        """
        try:
            llm = self._get_llm()
            response = llm.chat_complete(
                prompt=self._PROMPT.format(query=query, n=self.n),
                system="Bạn là expert query rewriting. Chỉ trả về JSON array thuần, không thêm gì.",
                max_tokens=200,
                temperature=0.5,
            )
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            expanded = json.loads(response)
            if not isinstance(expanded, list):
                return []

            result = [q.strip() for q in expanded if isinstance(q, str) and q.strip()]
            logger.info(f"[QueryExpander] {len(result)} variants for: '{query[:50]}'")
            return result[: self.n]

        except Exception as e:
            logger.warning(f"[QueryExpander] Failed, using original query only: {e}")
            return []
