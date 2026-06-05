from typing import List, Optional

class PromptBuilder:
    """Class điều phối việc xây dựng Prompt linh hoạt cho LLM."""

    @staticmethod
    def build_system_prompt(mode: str = "standard") -> str:
        """Tạo System Prompt dựa trên chế độ trả lời."""
        base_rules = (
            "Bạn là chuyên gia phân tích Video. NHIỆM VỤ: Trả lời CHỈ dựa trên [Tài liệu], [Tổng quan] và [Bản đồ thực thể] được cung cấp.\n"
            "QUY TẮC NGHIÊM NGẶT:\n"
            "1. Xưng 'tôi'. TUYỆT ĐỐI KHÔNG dùng câu mở đầu kiểu 'Tôi sẽ trả lời...', 'Dựa trên tài liệu...', 'Câu hỏi của bạn là...'. Bắt đầu NGAY bằng nội dung trả lời.\n"
            "2. LUÔN trích dẫn mốc [mm:ss] cho MỌI thông tin cụ thể lấy từ tài liệu.\n"
            "3. Nếu câu hỏi về số lượng, ưu tiên dùng số liệu từ [Bản đồ thực thể].\n"
            "4. TUYỆT ĐỐI KHÔNG bịa thêm thông tin không có trong [Tài liệu]. "
            "Nếu thông tin không xuất hiện trong tài liệu, trả lời: "
            "'Tôi không tìm thấy thông tin này trong video.'\n"
            "5. Chỉ trích dẫn [mm:ss] nếu timestamp đó thực sự xuất hiện trong [Tài liệu] bên dưới.\n"
        )

        if mode == "mindmap":
            return base_rules + (
                "6. ĐỊNH DẠNG: Sử dụng Mermaid Syntax (graph TD) để vẽ sơ đồ quan hệ nếu cần thiết.\n"
                "7. Sử dụng bảng (Markdown Table) cho các dữ liệu so sánh."
            )

        return base_rules + "6. Trả lời bằng tiếng Việt chuyên sâu, chính xác."

    @staticmethod
    def build_user_prompt(
        query: str, 
        context_str: str, 
        global_summary: Optional[str] = None, 
        graph_facts: Optional[List[str]] = None,
        graph_summary: Optional[str] = None,
        chat_history: str = ""
    ) -> str:
        """Lắp ghép các thành phần dữ liệu thành User Prompt hoàn chỉnh."""
        
        summary_info = f"\n--- TỔNG QUAN VIDEO ---\n{global_summary}\n" if global_summary else ""
        graph_info = f"\n--- BẢN ĐỒ THỰC THỂ TOÀN CỤC ---\n{graph_summary}\n" if graph_summary else ""
        
        facts_str = ""
        if graph_facts:
            facts_str = "\n--- DỮ LIỆU ĐỒ THỊ (TRỰC QUAN) ---\n" + "\n".join([f"- {f}" for f in graph_facts]) + "\n"

        return f"""
Câu hỏi từ người dùng: "{query}"

{chat_history}
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
        return f"""Bạn là một kiểm toán viên AI nghiêm khắc. Kiểm tra câu trả lời sau theo từng bước.

Câu hỏi: {query}
Câu trả lời cần kiểm tra:
{draft_answer}

Sự thật đã xác minh từ Knowledge Graph:
{facts_block}

NHIỆM VỤ KIỂM TRA:
1. Tìm mọi thông tin trong câu trả lời MÂU THUẪN với Graph Facts → sửa lại.
2. Tìm mọi thông tin trong câu trả lời KHÔNG CÓ TRONG Graph Facts VÀ không có [mm:ss] citation → xóa hoặc đánh dấu "không xác nhận được".
3. Giữ nguyên cấu trúc, văn phong và mọi [mm:ss] hợp lệ của câu trả lời gốc.
4. Chỉ trả về bản câu trả lời cuối cùng đã được sửa, không giải thích thêm."""
