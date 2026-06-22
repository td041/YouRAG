import re
from typing import List, Dict, Any, Optional
from src.engine.generation.llm_client import LLMClient
from src.engine.generation.prompt_builder import PromptBuilder
from src.core.logger import logger
from src.core.utils import format_timestamp

_NO_CONTEXT_REPLY = "Tôi không tìm thấy thông tin liên quan đến câu hỏi này trong video."

# Unicode superscript digits/symbols that LLMs sometimes emit as footnote markers
_SUPERSCRIPT_RE = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰⁻⁺⁼⁽⁾ⁿ]+")


class AnswerGenerator:
    """Module cuối cùng của luồng RAG: Sinh câu trả lời dựa trên ngữ cảnh đã được truy xuất.
    Nhận dữ liệu (Chunks) từ Reranker + Câu hỏi của người dùng -> Xây dựng Prompt -> Gọi LLM.
    """

    def __init__(self):
        self.llm = LLMClient()

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
            ts_end = format_timestamp(end)

            context_parts.append(
                f"[Tài liệu {i}] Mốc: {ts_start} - {ts_end}{context_prefix}\n"
                f"Nội dung: {raw_text}\n"
            )

        return "\n".join(context_parts)

    def _strip_superscripts(self, text: str) -> str:
        """Strip Unicode superscript footnote markers hallucinated by the LLM (¹²³⁴...)."""
        cleaned = _SUPERSCRIPT_RE.sub("", text)
        if cleaned != text:
            logger.warning("[CitationGrounding] Stripped superscript footnote markers from answer")
        return cleaned

    def _validate_citations(self, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Citation Grounding: xóa [mm:ss] không tương ứng với bất kỳ chunk nào đã retrieve.

        Mỗi [mm:ss] trong câu trả lời phải rơi vào khoảng [start_time, end_time + 10s]
        của ít nhất một chunk. Nếu không → citation bịa, xóa đi.
        """
        if not retrieved_chunks:
            return answer

        valid_ranges: List[tuple] = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            try:
                start = float(meta.get("start_time", 0))
                end = float(meta.get("end_time", 0))
                valid_ranges.append((start, end))
            except (TypeError, ValueError):
                pass

        if not valid_ranges:
            return answer

        def ts_to_secs(ts: str) -> float:
            parts = ts.split(":")
            try:
                if len(parts) == 3:  # H:MM:SS
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                if len(parts) == 2:  # M:SS or MM:SS
                    return int(parts[0]) * 60 + float(parts[1])
                return float(ts)
            except ValueError:
                return -1.0

        def is_grounded(ts: str) -> bool:
            t = ts_to_secs(ts)
            if t < 0:
                return True  # không parse được → giữ nguyên
            return any(start - 5 <= t <= end + 10 for start, end in valid_ranges)

        # Match [M:SS], [MM:SS], [H:MM:SS] formats
        citations = re.findall(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]', answer)
        invalid = [c for c in citations if not is_grounded(c)]

        if invalid:
            logger.warning(f"[CitationGrounding] Removing {len(invalid)} ungrounded citations: {invalid}")
            for ts in invalid:
                answer = answer.replace(f"[{ts}]", "")

        return answer.strip()

    def _build_prompts(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        global_summary: Optional[str],
        graph_facts: Optional[List[str]],
        graph_summary: Optional[str],
        chat_history: str,
    ) -> tuple[str, str]:
        context_str = self._build_context_string(retrieved_chunks)
        mode = "mindmap" if any(k in query.lower() for k in ["quan hệ", "cấu trúc", "sơ đồ", "bản đồ"]) else "standard"
        system_prompt = PromptBuilder.build_system_prompt(mode=mode)

        # Build explicit allowlist of valid timestamps so streaming path
        # can't produce hallucinated citations (validation runs post-stream).
        seen: set = set()
        valid_timestamps: List[str] = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            try:
                ts = format_timestamp(float(meta.get("start_time", 0)))
                if ts not in seen:
                    seen.add(ts)
                    valid_timestamps.append(f"[{ts}]")
            except (TypeError, ValueError):
                pass

        user_prompt = PromptBuilder.build_user_prompt(
            query=query,
            context_str=context_str,
            global_summary=global_summary,
            graph_facts=graph_facts,
            graph_summary=graph_summary,
            chat_history=chat_history,
            valid_timestamps=valid_timestamps if valid_timestamps else None,
        )
        return system_prompt, user_prompt

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        global_summary: Optional[str] = None,
        graph_facts: Optional[List[str]] = None,
        graph_summary: Optional[str] = None,
        chat_history: str = "",
    ) -> str:
        """Thực thi sinh câu trả lời RAG với bối cảnh thực thể từ Knowledge Graph."""
        logger.info(f"✨ Production Generation (Graph Enhanced): '{query}'")

        if not retrieved_chunks:
            logger.warning("[AnswerGenerator] No chunks retrieved — returning no-context reply")
            return _NO_CONTEXT_REPLY

        system_prompt, user_prompt = self._build_prompts(
            query, retrieved_chunks, global_summary, graph_facts, graph_summary, chat_history
        )

        try:
            draft_answer = self.llm.chat_complete(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=2000,
                temperature=0.2,
            )

            # Self-Correction Step
            if graph_facts and draft_answer and "Lỗi:" not in draft_answer:
                logger.info("🔍 Kích hoạt Self-Correction với Graph Facts...")
                check_prompt = PromptBuilder.build_self_correction_prompt(query, draft_answer, graph_facts)
                final_answer = self.llm.chat_complete(
                    prompt=check_prompt,
                    system="Bạn là một kiểm toán viên nghiêm khắc. Sửa lỗi dựa trên Graph Facts.",
                    max_tokens=2000,
                    temperature=0.1,
                )
            else:
                final_answer = draft_answer

            # Strip hallucinated superscript footnotes then validate [mm:ss] citations
            final_answer = self._strip_superscripts(final_answer)
            final_answer = self._validate_citations(final_answer, retrieved_chunks)
            return final_answer

        except Exception as e:
            logger.error(f"❌ Lỗi Generation: {e}")
            return f"Lỗi: {str(e)}"

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        global_summary: Optional[str] = None,
        graph_facts: Optional[List[str]] = None,
        graph_summary: Optional[str] = None,
        chat_history: str = "",
    ):
        """True LLM streaming — graph_facts already injected into system_prompt so the model
        sees them on first pass. Self-correction (2 extra LLM calls) is reserved for the
        non-streaming generate() path used by batch evaluation.
        """
        logger.info(f"✨ Streaming (graph_facts={'yes' if graph_facts else 'no'}): '{query}'")

        if not retrieved_chunks:
            logger.warning("[AnswerGenerator] No chunks retrieved — returning no-context reply")
            yield _NO_CONTEXT_REPLY
            return

        system_prompt, user_prompt = self._build_prompts(
            query, retrieved_chunks, global_summary, graph_facts, graph_summary, chat_history
        )

        try:
            accumulated = []
            for chunk in self.llm.chat_complete_stream(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=2000,
                temperature=0.2,
            ):
                accumulated.append(chunk)
                yield chunk

            # Post-stream citation grounding — logs only, answer already sent
            full_text = "".join(accumulated)
            validated = self._validate_citations(full_text, retrieved_chunks)
            if validated != full_text:
                logger.warning("[CitationGrounding] Stream had ungrounded citations (already sent to client)")

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield f"\n[Error: {str(e)}]"
