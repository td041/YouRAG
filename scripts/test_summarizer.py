import sys
import os

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.database import db_instance
from src.engine.generation.summarizer import VideoSummarizer

def test_summary():
    print(f"==================================================")
    print(f"🎬 KÍCH HOẠT TÓM TẮT VIDEO")
    print(f"==================================================")

    cols = db_instance.client.list_collections()
    if not cols:
        print("❌ Lỗi: Không có video nào trong Qdrant. Hãy chạy file test_ingestion.py trước.")
        return

    # Lấy video đầu tiên trong DB ra test
    first_col = cols[0].name
    print(f"📁 Video mục tiêu: {first_col}\n")
    print(f"⏳ Đang yêu cầu AI Đọc & Suy luận (Giai đoạn này mất khoảng 5-15 giây). Vui lòng đợi...\n")

    summarizer = VideoSummarizer()
    # Chạy Tóm tắt
    result = summarizer.summarize(first_col)

    if not result:
        print("⚠️ Không thể tạo bản tóm tắt.")
        return

    print("✅ KẾT QUẢ TÓM TẮT TỪ AI: \n")
    print(result)
    print("\n" + "="*60)

if __name__ == "__main__":
    test_summary()
