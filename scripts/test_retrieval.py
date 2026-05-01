import sys
import os

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.database import db_instance
from src.engine.retrieval.hybrid_search import HybridRetriever
from src.engine.ranking.cross_encoder import CrossEncoderReranker

import time

def typewriter_print(text: str, delay: float = 0.015):
    """Hiệu ứng gõ chữ mượt mà cho Terminal."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # Xuống dòng cuối cùng

def test_retrieval(query: str):
    typewriter_print(f"\n==================================================")
    typewriter_print(f"🎯 CÂU HỎI THỬ NGHIỆM: '{query}'")
    typewriter_print(f"==================================================")

    cols = db_instance.client.get_collections().collections
    if not cols:
        typewriter_print("❌ Lỗi: Không có video/dữ liệu nào trong Qdrant. Hãy chạy ingestion trước.")
        return

    first_col = cols[0].name
    typewriter_print(f"📁 Truy vấn trong bộ dữ liệu (Collection): {first_col}\n")

    # Sử dụng Hybrid Search (kết hợp Dense + Sparse)
    hybrid = HybridRetriever(top_k=3) # Lấy 3 kết quả tốt nhất
    results = hybrid.search(query, collection_name=first_col)

    # Khởi tạo mô hình Cross-Encoder để Reranking
    reranker = CrossEncoderReranker()
    final_results = reranker.rerank(query=query, chunks=results, top_k=2)

    typewriter_print("🏅 KẾT QUẢ TOP 2 (Đã qua Reranker Lọc Phễu): \n")
    for i, res in enumerate(final_results, 1):
        typewriter_print(f"[{i}] Hybrid Score: {res.get('hybrid_score', 0):.4f} | Rerank Score: {res.get('rerank_score', 0):.4f} | ID: {res['id']}")
        
        # In ra bối cảnh sinh bởi LLM (từ metadata)
        context = res['metadata'].get('context_snippet', 'Không có context snippet')
        typewriter_print(f"   🧠 Ngữ cảnh LLM: {context}")
        
        # In ra thời gian gốc trong video
        start = res['metadata'].get('start_time', 0)
        end = res['metadata'].get('end_time', 0)
        typewriter_print(f"   ⏱️ Thời điểm trong video: {start}s - {end}s")
        
        typewriter_print("\n   📜 Nội dung Cắt (Chunk):")
        # In ra 300 ký tự đầu tiên để xem
        content = res['content']
        content_preview = content[:300] + ("..." if len(content) > 300 else "")
        typewriter_print(f"      {content_preview}", delay=0.005) # Chạy chữ nhanh hơn cho nội dung
        typewriter_print("-" * 60)

if __name__ == "__main__":
    # Test thử 2 câu hỏi từ dễ đến khó về luật chơi Bomb busters
    questions = [
        "How do I win?",
        "Can I talk to my teammates?"
    ]
    
    for q in questions:
        test_retrieval(q)
        print("\n")
