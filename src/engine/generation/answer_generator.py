from typing import List, Dict, Any, Optional
from src.engine.generation.llm_client import LLMClient
from src.engine.generation.prompt_builder import PromptBuilder
from src.cache.semantic_cache import SemanticCache
from src.core.logger import logger
from src.core.utils import format_timestamp

class AnswerGenerator:
    """Module cuối cùng của luồng RAG: Sinh câu trả lời dựa trên ngữ cảnh đã được truy xuất.
    Nhận dữ liệu (Chunks) từ Reranker + Câu hỏi của người dùng -> Xây dựng Prompt -> Gọi LLM.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.cache = SemanticCache()
        
    def _build_context_string(self, chunks: List[Dict[str, Any]]) -> str:
        """Đóng gói các đoạn tài liệu thành một định dạng văn bản dễ đọc cho LLM."""
        if not chunks:
            return "Không có thông tin nào được tìm thấy trong video."
            
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            start = metadata.get("start_time", 0)
            end = metadata.get("end_time", 0)
            
            snippet = metadata.get("context_snippet", "")
            context_prefix = f" (Ngữ cảnh: {snippet})" if snippet else ""
            
            raw_text = chunk["content"]
            if "\n\n" in raw_text:
                raw_text = raw_text.split("\n\n")[-1]

            ts_start = format_timestamp(start)
            ts_end   = format_timestamp(end)

            context_parts.append(
                f"[Tài liệu {i}] Mốc: {ts_start} - {ts_end}{context_prefix}\n"
                f"Nội dung: {raw_text}\n"
            )
            
        return "\n".join(context_parts)

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]], global_summary: Optional[str] = None, graph_facts: Optional[List[str]] = None, graph_summary: Optional[str] = None, chat_history: str = "") -> str:
        """Thực thi sinh câu trả lời RAG với bối cảnh thực thể từ Knowledge Graph."""
        logger.info(f"✨ Production Generation (Graph Enhanced): '{query}'")

        cached = self.cache.check_cache(query)
        if cached is not None:
            logger.info(f"[AnswerGenerator] Cache hit for query: '{query[:60]}'")
            return cached["answer"]

        context_str = self._build_context_string(retrieved_chunks)
        
        # Quyết định mode
        mode = "mindmap" if any(k in query.lower() for k in ["quan hệ", "cấu trúc", "sơ đồ", "bản đồ"]) else "standard"
        
        system_prompt = PromptBuilder.build_system_prompt(mode=mode)
        user_prompt = PromptBuilder.build_user_prompt(
            query=query,
            context_str=context_str,
            global_summary=global_summary,
            graph_facts=graph_facts,
            graph_summary=graph_summary,
            chat_history=chat_history
        )
        
        try:
            draft_answer = self.llm.chat_complete(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=2000,
                temperature=0.2
            )
            
            # Self-Correction Step
            if graph_facts and draft_answer and "Lỗi:" not in draft_answer:
                logger.info("🔍 Kích hoạt Self-Correction với Graph Facts...")
                check_prompt = PromptBuilder.build_self_correction_prompt(query, draft_answer, graph_facts)
                final_answer = self.llm.chat_complete(
                    prompt=check_prompt,
                    system="Bạn là một kiểm toán viên nghiêm khắc. Sửa lỗi dựa trên Graph Facts.",
                    max_tokens=2000,
                    temperature=0.1
                )
                self.cache.save_to_cache(query, final_answer)
                return final_answer

            self.cache.save_to_cache(query, draft_answer)
            return draft_answer

        except Exception as e:
            logger.error(f"❌ Lỗi Generation: {e}")
            return f"Lỗi: {str(e)}"

    def generate_stream(self, query: str, retrieved_chunks: List[Dict[str, Any]], global_summary: Optional[str] = None, graph_facts: Optional[List[str]] = None, graph_summary: Optional[str] = None, chat_history: str = ""):
        """Streaming với bối cảnh thực thể từ Knowledge Graph."""
        logger.info(f"✨ Production Streaming (Graph Enhanced): '{query}'")
        
        context_str = self._build_context_string(retrieved_chunks)
        
        # Quyết định mode
        mode = "mindmap" if any(k in query.lower() for k in ["quan hệ", "cấu trúc", "sơ đồ", "bản đồ"]) else "standard"
        
        system_prompt = PromptBuilder.build_system_prompt(mode=mode)
        user_prompt = PromptBuilder.build_user_prompt(
            query=query,
            context_str=context_str,
            global_summary=global_summary,
            graph_facts=graph_facts,
            graph_summary=graph_summary,
            chat_history=chat_history
        )
        
        try:
            for chunk in self.llm.chat_complete_stream(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=2000,
                temperature=0.2
            ):
                yield chunk
        except Exception as e:
            logger.error(f"❌ Lỗi Streaming: {e}")
            yield f"\n[Lỗi: {str(e)}]"
