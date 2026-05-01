import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.database import db_instance
from src.engine.retrieval.hybrid_search import HybridRetriever
from src.engine.retrieval.cross_encoder import CrossEncoderReranker
from src.engine.generation.answer_generator import AnswerGenerator

def ask(query: str, collection_name: str):
    """Chạy toàn bộ pipeline RAG: Hybrid Search -> Reranker -> Generation."""
    print(f"\n{'='*60}")
    print(f"❓ CÂU HỎI: {query}")
    print(f"{'='*60}")

    # Bước 1: Hybrid Search (Dense + Sparse + RRF)
    hybrid = HybridRetriever(top_k=6)  # Lấy Top 6 làm đầu vào cho Reranker
    candidates = hybrid.search(query, collection_name=collection_name)

    if not candidates:
        print("⚠️  Không tìm thấy tài liệu nào liên quan.")
        return

    # Bước 2: Cross-Encoder Reranker (Lọc Phễu)
    reranker = CrossEncoderReranker()
    final_chunks = reranker.rerank(query=query, chunks=candidates, top_k=3)

    # Bước 3: Answer Generation (Sinh câu trả lời)
    generator = AnswerGenerator()
    answer = generator.generate(query=query, retrieved_chunks=final_chunks)

    print(f"\n💬 CÂU TRẢ LỜI:\n")
    print(answer)
    print(f"\n{'─'*60}")


if __name__ == "__main__":
    cols = db_instance.client.list_collections()
    if not cols:
        print("❌ Không có dữ liệu trong DB. Hãy chạy scripts/test_ingestion.py trước!")
        sys.exit(1)

    col_name = cols[0].name
    print(f"✅ Sử dụng Collection: {col_name}")

    # Thử nhiều câu hỏi, từ dễ đến khó
    questions = [
        "How do players win the game?",
        "What happens when someone cuts the wrong wire?",
        "How many players can play this game?",
    ]

    # Khởi tải Reranker 1 lần duy nhất để đỡ tải lại model nhiều lần
    reranker = CrossEncoderReranker()

    for q in questions:
        hybrid = HybridRetriever(top_k=6)
        candidates = hybrid.search(q, collection_name=col_name)
        final_chunks = reranker.rerank(query=q, chunks=candidates, top_k=3)
        generator = AnswerGenerator()
        answer = generator.generate(query=q, retrieved_chunks=final_chunks)
        
        print(f"\n{'='*60}")
        print(f"❓ CÂU HỎI: {q}")
        print(f"{'='*60}")
        print(f"\n💬 CÂU TRẢ LỜI:\n{answer}")
        print(f"{'─'*60}")
