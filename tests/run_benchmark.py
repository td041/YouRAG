"""
Script chạy Performance Benchmark & RAGAS Evaluation.

Sử dụng:
    # Bước 1: Tạo bộ câu hỏi đánh giá
    poetry run python tests/run_benchmark.py --generate

    # Bước 2: Chạy đánh giá và so sánh (tự động log vào MLflow)
    poetry run python tests/run_benchmark.py --evaluate

    # Xem kết quả MLflow
    # http://localhost:5001
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.database import db_instance
from src.core.config import settings
from src.core.logger import logger


def get_first_collection() -> str:
    """Lấy collection đầu tiên trong Qdrant."""
    cols = db_instance.client.get_collections().collections
    if not cols:
        logger.error("❌ Database Qdrant trống! Hãy ingest video trước.")
        sys.exit(1)
    return cols[0].name


def _try_mlflow():
    """Trả về mlflow module nếu có, None nếu không."""
    try:
        import mlflow
        return mlflow
    except ImportError:
        logger.warning("[MLflow] Không tìm thấy mlflow — pip install mlflow. Bỏ qua logging.")
        return None


def run_generate(collection_name: str, num_questions: int = 10):
    """Bước 1: Sinh bộ câu hỏi đánh giá."""
    from src.engine.evaluation.dataset_generator import EvalDatasetGenerator

    gen = EvalDatasetGenerator(num_questions=num_questions)
    dataset = gen.generate(collection_name)

    logger.info(f"\n📋 Tổng kết: Đã sinh {len(dataset)} câu hỏi đánh giá.")
    for i, item in enumerate(dataset, 1):
        logger.info(f"   [{i}] {item['question']}")


def run_evaluate(collection_name: str):
    """Bước 2: Chạy RAGAS Evaluation so sánh Baseline vs Advanced, log vào MLflow."""
    from src.engine.evaluation.ragas_evaluator import RAGASEvaluator

    evaluator = RAGASEvaluator()
    comparison = evaluator.compare_baseline_vs_advanced(collection_name)

    naive_s  = comparison.get("0_naive",    {}).get("ragas_scores", {})
    hybrid_s = comparison.get("1_hybrid",   {}).get("ragas_scores", {})
    adv_s    = comparison.get("2_advanced", {}).get("ragas_scores", {})

    core_metrics = [
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall", "factual_correctness",
    ]

    logger.info("\n🏆 KẾT LUẬN:")
    for metric in core_metrics:
        n = naive_s.get(metric)
        h = hybrid_s.get(metric)
        a = adv_s.get(metric)
        if n is None:
            continue
        scores = {k: v for k, v in [("Naive", n), ("Hybrid", h), ("Advanced", a)] if v is not None}
        best_tier = max(scores, key=scores.get)
        best_val  = scores[best_tier]
        icon = "✅" if best_val > n else "⚪"
        logger.info(
            f"   {icon} {metric}: tốt nhất là {best_tier} ({best_val:.4f}) | "
            f"Naive={n:.4f} Hybrid={h:.4f if h else 'N/A'} Advanced={a:.4f if a else 'N/A'}"
        )

    # ── MLflow logging ────────────────────────────────────────────────────
    mlflow = _try_mlflow()
    if mlflow:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("YouRAG RAGAS Benchmark")

        tier_data = [
            ("Naive",    naive_s,  comparison.get("0_naive",    {})),
            ("Hybrid",   hybrid_s, comparison.get("1_hybrid",   {})),
            ("Advanced", adv_s,    comparison.get("2_advanced", {})),
        ]
        for tier_name, scores, tier_meta in tier_data:
            if not scores:
                continue
            with mlflow.start_run(run_name=f"{tier_name} — {collection_name[:40]}"):
                # Params
                mlflow.log_params({
                    "tier": tier_name,
                    "collection": collection_name,
                    "llm_model": settings.LLM_MODEL_NAME,
                    "embedding_model": settings.EMBEDDING_MODEL_NAME,
                    "reranker_model": settings.CROSS_ENCODER_MODEL,
                    "top_k_retrieval": settings.TOP_K_RETRIEVAL,
                    "top_k_rerank": settings.TOP_K_RERANK,
                    "chunk_size": settings.CHUNK_SIZE,
                })
                # RAGAS metrics
                for metric in core_metrics:
                    if scores.get(metric) is not None:
                        mlflow.log_metric(metric, scores[metric])
                # Latency
                latency = tier_meta.get("latency_mean")
                if latency is not None:
                    mlflow.log_metric("latency_mean_s", latency)
                # Composite score
                composite = sum(
                    scores[m] for m in core_metrics if scores.get(m) is not None
                ) / max(1, sum(1 for m in core_metrics if scores.get(m) is not None))
                mlflow.log_metric("composite_score", composite)

        logger.info(f"✅ MLflow: Đã log {len(tier_data)} runs → {mlflow_uri}")


def main():
    parser = argparse.ArgumentParser(description="YouRAG RAGAS Evaluation Pipeline")
    parser.add_argument("--generate", action="store_true", help="Sinh bộ câu hỏi đánh giá")
    parser.add_argument("--evaluate", action="store_true", help="Chạy RAGAS evaluation")
    parser.add_argument("--all",      action="store_true", help="Chạy toàn bộ pipeline")
    parser.add_argument("--collection",    type=str, default=None, help="Tên Collection Qdrant")
    parser.add_argument("--num-questions", type=int, default=10,   help="Số câu hỏi cần sinh (mặc định: 10)")

    args = parser.parse_args()

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
        print("\n📊 Xem MLflow: http://localhost:5001")


if __name__ == "__main__":
    main()
