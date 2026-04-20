"""
Script chạy Performance Benchmark & RAGAS Evaluation.

Sử dụng:
    # Bước 1: Tạo bộ câu hỏi đánh giá
    poetry run python tests/run_benchmark.py --generate

    # Bước 2: Chạy đánh giá và so sánh
    poetry run python tests/run_benchmark.py --evaluate
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.database import db_instance
from src.core.logger import logger


def get_first_collection() -> str:
    """Lấy collection đầu tiên trong Qdrant."""
    cols = db_instance.client.get_collections().collections
    if not cols:
        logger.error("❌ Database Qdrant trống! Hãy ingest video trước.")
        sys.exit(1)
    return cols[0].name


def run_generate(collection_name: str, num_questions: int = 10):
    """Bước 1: Sinh bộ câu hỏi đánh giá."""
    from src.engine.evaluation.dataset_generator import EvalDatasetGenerator
    
    gen = EvalDatasetGenerator(num_questions=num_questions)
    dataset = gen.generate(collection_name)
    
    logger.info(f"\n📋 Tổng kết: Đã sinh {len(dataset)} câu hỏi đánh giá.")
    for i, item in enumerate(dataset, 1):
        logger.info(f"   [{i}] {item['question']}")


def run_evaluate(collection_name: str):
    """Bước 2: Chạy RAGAS Evaluation so sánh Baseline vs Advanced."""
    from src.engine.evaluation.ragas_evaluator import RAGASEvaluator
    
    evaluator = RAGASEvaluator()
    comparison = evaluator.compare_baseline_vs_advanced(collection_name)
    
    logger.info("\n🏆 KẾT LUẬN:")
    improvements = comparison.get("improvements", {})
    for metric, data in improvements.items():
        if isinstance(data, dict) and data.get("delta", 0) > 0:
            logger.info(f"   ✅ {metric}: Tăng +{data['improvement_pct']}%")
        elif isinstance(data, dict):
            logger.info(f"   ⚠️ {metric}: {data.get('delta', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="YouRAG RAGAS Evaluation Pipeline")
    parser.add_argument("--generate", action="store_true", help="Sinh bộ câu hỏi đánh giá")
    parser.add_argument("--evaluate", action="store_true", help="Chạy RAGAS evaluation")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ pipeline")
    parser.add_argument("--collection", type=str, default=None, help="Tên Collection Qdrant")
    parser.add_argument("--num-questions", type=int, default=10, help="Số câu hỏi cần sinh (mặc định: 10)")
    
    args = parser.parse_args()
    
    # Xác định collection
    collection = args.collection or get_first_collection()
    logger.info(f"📂 Collection: {collection}")
    
    if args.all or (args.generate and args.evaluate):
        run_generate(collection, args.num_questions)
        run_evaluate(collection)
    elif args.generate:
        run_generate(collection, args.num_questions)
    elif args.evaluate:
        run_evaluate(collection)
    else:
        parser.print_help()
        print("\n💡 Ví dụ nhanh:")
        print("   poetry run python tests/run_benchmark.py --all")
        print("   poetry run python tests/run_benchmark.py --evaluate")


if __name__ == "__main__":
    main()
