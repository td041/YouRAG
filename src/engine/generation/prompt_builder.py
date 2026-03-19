from typing import List, Optional

class PromptBuilder:
    """Class điều phối việc xây dựng Prompt linh hoạt cho LLM."""

    @staticmethod
    def build_system_prompt(mode: str = "standard") -> str:
        """Tạo System Prompt dựa trên chế độ trả lời."""
        base_rules = (
            "Bạn là chuyên gia phân tích Video. NHIỆM VỤ: Trả lời dựa trên [Tài liệu], [Tổng quan] và [Bản đồ thực thể].\n"
            "QUY TẮC:\n"
            "1. Xưng 'tôi'. KHÔNG mở đầu rườm rà. Hãy vào thẳng nội dung.\n"
            "2. LUÔN trích dẫn mốc [mm:ss] cho thông tin từ tài liệu.\n"
            "3. Nếu câu hỏi về số lượng, hãy ưu tiên dùng số liệu từ [Bản đồ thực thể].\n"
        )
        
        if mode == "mindmap":
            return base_rules + (
                "4. ĐỊNH DẠNG: Sử dụng Mermaid Syntax (graph TD) để vẽ sơ đồ quan hệ nếu cần thiết.\n"
                "5. Sử dụng bảng (Markdown Table) cho các dữ liệu so sánh."
            )
            
        return base_rules + "4. Trả lời bằng tiếng Việt chuyên sâu, chính xác."

    @staticmethod
    def build_user_prompt(
        query: str, 
        context_str: str, 
        global_summary: Optional[str] = None, 
        graph_facts: Optional[List[str]] = None,
        graph_summary: Optional[str] = None
    ) -> str:
        """Lắp ghép các thành phần dữ liệu thành User Prompt hoàn chỉnh."""
        
        summary_info = f"\n--- TỔNG QUAN VIDEO ---\n{global_summary}\n" if global_summary else ""
        graph_info = f"\n--- BẢN ĐỒ THỰC THỂ TOÀN CỤC ---\n{graph_summary}\n" if graph_summary else ""
        
        facts_str = ""
        if graph_facts:
            facts_str = "\n--- DỮ LIỆU ĐỒ THỊ (TRỰC QUAN) ---\n" + "\n".join([f"- {f}" for f in graph_facts]) + "\n"

        return f"""
Câu hỏi từ người dùng: "{query}"

{summary_info}
{graph_info}
{facts_str}

--- TÀI LIỆU CHI TIẾT (VIDEO CHUNKS) ---
{context_str}

Hãy thực hiện nhiệm vụ phân tích và trả lời chính xác nhất.
"""

    @staticmethod
    def build_self_correction_prompt(query: str, draft_answer: str, graph_facts: List[str]) -> str:
        """Tạo prompt để AI tự kiểm tra lại câu trả lời (Self-Correction)."""
        facts_block = "\n".join([f"- {f}" for f in graph_facts])
        return f"""
Bạn là một kiểm toán viên AI. Hãy kiểm tra câu trả lời sau có mâu thuẫn với sự thật trong Đồ thị tri thức không.

Câu hỏi: {query}
Câu trả lời dự kiến: {draft_answer}

Sự thật từ Đồ thị (Graph Facts):
{facts_block}

NHIỆM VỤ:
- Nếu có mâu thuẫn, hãy sửa lại câu trả lời để khớp với Graph Facts.
- Nếu câu trả lời thiếu mốc thời gian [mm:ss] cho các thông tin quan trọng, hãy bổ sung (nếu có trong draft).
- Chỉ trả về bản câu trả lời đã được sửa đổi cuối cùng.
"""
