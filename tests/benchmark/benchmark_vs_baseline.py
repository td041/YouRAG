import json
import os
import sys
import time

# Đảm bảo import được code dự án
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.database import db_instance
from src.engine.retrieval.dense_search import DenseRetriever
from src.engine.retrieval.hybrid_search import HybridRetriever
from src.engine.retrieval.cross_encoder import CrossEncoderReranker
from src.core.logger import logger

def load_dataset(filepath="tests/benchmark/dataset.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_benchmark():
    try:
        dataset = load_dataset()
    except FileNotFoundError:
        logger.error("❌ Không tìm thấy file dataset.json!")
        return

    logger.info("==============================================")
    logger.info("   🚀 BẮT ĐẦU CHẠY BENCHMARK ĐÁNH GIÁ (EVAL) ")
    logger.info("==============================================\n")

    cols = db_instance.client.get_collections().collections
    if not cols:
        logger.error("❌ Database Qdrant trống không! Hãy cào 1 video (Ingest) trước.")
        return
    
    collection_name = cols[0].name
    logger.info(f"📂 Đang thử nghiệm trên bộ dữ liệu: {collection_name}")

    # --- KHỞI TẠO 2 ENGINE ĐỂ "THI ĐẤU" ---
    
    # 1. NAIVE RAG (Baseline Cơ Bản): Chỉ dùng Dense Vector, lấy luôn Top 3
    naive_retriever = DenseRetriever(top_k=3)
    
    # 2. ADVANCED SOTA RAG (Kiến trúc của bạn): Hybrid (Top 5) + Reranker (Lọc về Top 3)
    hybrid_retriever = HybridRetriever(top_k=5)
    reranker = CrossEncoderReranker()
    
    total_queries = len(dataset)
    
    # Bảng điểm
    stats = {
        "naive": {"hit": 0, "mrr": 0.0, "time": 0.0},
        "sota": {"hit": 0, "mrr": 0.0, "time": 0.0}
    }

    for idx, item in enumerate(dataset, 1):
        query = item["query"]
        expected_range = item["ground_truth_time"]
        
        logger.info(f"\n[Câu {idx}/{total_queries}] Query: '{query}' (Target: {expected_range[0]}s - {expected_range[1]}s)")
        
        # ==========================================
        # 🏃‍♂️ VÒNG 1: NAIVE RAG (CƠ BẢN) thi đấu
        # ==========================================
        t0 = time.time()
        naive_results = naive_retriever.search(query, collection_name=collection_name)
        stats["naive"]["time"] += (time.time() - t0)
        
        # Chấm điểm Naive
        for i, res in enumerate(naive_results, 1):
            start = res['metadata'].get('start_time', 0)
            end = res['metadata'].get('end_time', 0)
            if (start <= expected_range[1]) and (end >= expected_range[0]):
                stats["naive"]["hit"] += 1
                stats["naive"]["mrr"] += 1.0 / i
                logger.info(f"   [Naive] 🟡 Trúng đích ở vị trí #{i}")
                break
        
        # ==========================================
        # 🚀 VÒNG 2: ADVANCED SOTA RAG thi đấu
        # ==========================================
        t0 = time.time()
        hybrid_candidates = hybrid_retriever.search(query, collection_name=collection_name)
        sota_results = reranker.rerank(query=query, chunks=hybrid_candidates, top_k=3)
        stats["sota"]["time"] += (time.time() - t0)
        
        # Chấm điểm SOTA
        for i, res in enumerate(sota_results, 1):
            start = res['metadata'].get('start_time', 0)
            end = res['metadata'].get('end_time', 0)
            if (start <= expected_range[1]) and (end >= expected_range[0]):
                stats["sota"]["hit"] += 1
                stats["sota"]["mrr"] += 1.0 / i
                logger.info(f"   [SOTA]  🟢 Trúng đích ở vị trí #{i}")
                break

    # -------- XUẤT BẢNG BÁO CÁO SO SÁNH ---------
    logger.info("\n==============================================")
    logger.info("   📈 BÁO CÁO: NAIVE BASELINE vs SOTA ADVANCED ")
    logger.info("==============================================")
    
    naive_hit_rate = round((stats["naive"]["hit"] / total_queries) * 100, 2)
    sota_hit_rate = round((stats["sota"]["hit"] / total_queries) * 100, 2)
    
    naive_mrr = round(stats["naive"]["mrr"] / total_queries, 2)
    sota_mrr = round(stats["sota"]["mrr"] / total_queries, 2)
    
    naive_latency = round(stats["naive"]["time"] / total_queries, 3)
    sota_latency = round(stats["sota"]["time"] / total_queries, 3)

    logger.info("1. Tỉ lệ trả lời đúng (Hit Rate @3):")
    logger.info(f"   - Naive (Baseline) : {naive_hit_rate}%")
    logger.info(f"   - YouRAG SOTA      : {sota_hit_rate}% (Tăng +{round(sota_hit_rate - naive_hit_rate, 2)}%) 🏆")
    
    logger.info("\n2. Độ chính xác xếp hạng (MRR Score):")
    logger.info(f"   - Naive (Baseline) : {naive_mrr}")
    logger.info(f"   - YouRAG SOTA      : {sota_mrr} (Tăng +{round(sota_mrr - naive_mrr, 2)}) 🏆")
    
    logger.info("\n3. Thời gian phản hồi (Latency/Query):")
    logger.info(f"   - Naive (Baseline) : {naive_latency}s ⚡")
    logger.info(f"   - YouRAG SOTA      : {sota_latency}s (Hy sinh {round(sota_latency - naive_latency, 3)}s để đổi lấy chất lượng)")
    logger.info("==============================================\n")

if __name__ == "__main__":
    run_benchmark()
