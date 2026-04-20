"""
RAGAS Evaluator - Đánh giá RAG Pipeline theo chuẩn nghiên cứu.

Tích hợp Framework RAGAS (Retrieval Augmented Generation Assessment)
để đo lường chất lượng toàn diện của hệ thống YouRAG.

Metrics đo đạc:
- Faithfulness: Câu trả lời có trung thành với context không?
- Answer Relevancy: Câu trả lời có liên quan đến câu hỏi không?
- Context Precision: Context được truy xuất có chính xác không?
- Context Recall: Context có bao phủ đủ ground truth không?
"""

import json
import os
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.database import db_instance
from src.core.config import settings
from src.core.logger import logger
from src.engine.retrieval.dense_search import DenseRetriever
from src.engine.retrieval.hybrid_search import HybridRetriever
from src.engine.ranking.cross_encoder import CrossEncoderReranker
from src.engine.generation.answer_generator import AnswerGenerator


class RAGASEvaluator:
    """Đánh giá End-to-End hệ thống RAG theo chuẩn RAGAS.
    
    Pipeline đánh giá:
    1. Load Evaluation Dataset (từ file JSON)
    2. Chạy RAG Pipeline (Retrieve → Rerank → Generate) cho từng câu hỏi
    3. Thu thập dữ liệu: question, answer, contexts, ground_truth
    4. Gọi RAGAS Framework tính toán metrics
    5. Xuất báo cáo chi tiết
    """

    def __init__(self):
        # Khởi tạo các Engine theo đúng pattern dự án
        self.hybrid_retriever = HybridRetriever(top_k=10)
        self.naive_retriever = DenseRetriever(top_k=3)
        self.reranker = CrossEncoderReranker()
        self.generator = AnswerGenerator()
        self.db = db_instance

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load bộ câu hỏi đánh giá từ file JSON."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _run_rag_pipeline(self, query: str, collection_name: str, mode: str = "advanced") -> Dict:
        """Chạy RAG pipeline cho 1 câu hỏi, trả về answer + contexts.
        
        Args:
            query: Câu hỏi
            collection_name: Collection Qdrant
            mode: 'naive' (Dense), 'hybrid_only' (Hybrid), hoặc 'advanced' (Hybrid+Rerank)
        """
        t0 = time.time()
        
        if mode == "naive":
            # Baseline: Dense Search thô, không Rerank
            results = self.naive_retriever.search(query, collection_name)
            contexts = [r["content"] for r in results]
        elif mode == "hybrid_only":
            # Cấp 1 SOTA: Hybrid Search (Chưa có Reranker)
            candidates = self.hybrid_retriever.search(query, collection_name)
            results = candidates[:4] # Cắt top 4 để so sánh công bằng
            contexts = [r["content"] for r in results]
        else:
            # Cấp 2 SOTA: Hybrid Search + Cross-Encoder Reranker (Advanced)
            candidates = self.hybrid_retriever.search(query, collection_name)
            results = self.reranker.rerank(query=query, chunks=candidates, top_k=4)
            contexts = [r["content"] for r in results]
        
        # Generate Answer
        answer = self.generator.generate(query=query, retrieved_chunks=results)
        
        latency = time.time() - t0
        
        return {
            "answer": answer,
            "contexts": contexts,
            "latency": round(latency, 3),
            "num_chunks": len(results)
        }

    def evaluate_with_ragas(
        self,
        collection_name: str,
        dataset_path: str = None,
        mode: str = "advanced"
    ) -> Dict[str, Any]:
        """Chạy đánh giá RAGAS đầy đủ.
        
        Args:
            collection_name: Tên Collection Qdrant
            dataset_path: Đường dẫn file eval_dataset.json
            mode: 'naive' hoặc 'advanced'
            
        Returns:
            Dict chứa các metrics RAGAS + chi tiết từng câu hỏi
        """
        if dataset_path is None:
            dataset_path = os.path.join(os.getcwd(), "tests", "benchmark", "eval_dataset.json")
        
        dataset = self._load_dataset(dataset_path)
        logger.info(f"🧪 RAGAS Evaluation [{mode.upper()}]: {len(dataset)} câu hỏi → [{collection_name}]")
        
        # Thu thập dữ liệu cho RAGAS
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        latencies = []
        per_question_results = []
        
        for idx, item in enumerate(dataset, 1):
            query = item["question"]
            gt = item["ground_truth"]
            
            logger.info(f"\n   [{idx}/{len(dataset)}] Q: {query[:60]}...")
            
            # Chạy RAG Pipeline
            result = self._run_rag_pipeline(query, collection_name, mode=mode)
            
            questions.append(query)
            answers.append(result["answer"])
            contexts_list.append(result["contexts"])
            ground_truths.append(gt)
            latencies.append(result["latency"])
            
            per_question_results.append({
                "question": query,
                "answer": result["answer"][:200] + "...",
                "ground_truth": gt[:200] + "...",
                "num_contexts": result["num_chunks"],
                "latency": result["latency"]
            })
            
            logger.info(f"   ✅ Answer: {result['answer'][:80]}... ({result['latency']}s)")
        
        # Tính metrics bằng RAGAS
        ragas_scores = self._compute_ragas_metrics(
            questions, answers, contexts_list, ground_truths
        )
        
        # Tổng hợp báo cáo
        avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0
        
        report = {
            "mode": mode,
            "total_questions": len(dataset),
            "avg_latency_s": avg_latency,
            "ragas_scores": ragas_scores,
            "per_question": per_question_results
        }
        
        return report

    def _compute_ragas_metrics(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Gọi RAGAS Framework để tính toán các metrics."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset

            # Chuẩn bị data theo format RAGAS
            eval_data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
            
            eval_dataset = Dataset.from_dict(eval_data)
            
            # Cấu hình LLM cho RAGAS (dùng Groq thông qua OpenAI-compatible)
            from langchain_openai import ChatOpenAI
            
            groq_key = settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else ""
            
            ragas_llm = ChatOpenAI(
                model="llama-3.1-8b-instant",
                openai_api_key=groq_key,
                openai_api_base="https://api.groq.com/openai/v1",
                temperature=0.0,
                max_tokens=1024,
            )

            # Chạy RAGAS Evaluation
            logger.info("\n🔬 Đang chạy RAGAS metrics (có thể mất vài phút)...")
            
            result = evaluate(
                dataset=eval_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=ragas_llm,
            )
            
            scores = {
                "faithfulness": round(result["faithfulness"], 4),
                "answer_relevancy": round(result["answer_relevancy"], 4),
                "context_precision": round(result["context_precision"], 4),
                "context_recall": round(result["context_recall"], 4),
            }
            
            logger.info(f"   📊 RAGAS Scores: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"❌ Lỗi RAGAS computation: {e}")
            logger.warning("⚠️ Fallback: Tính metrics thủ công (Hit Rate + MRR)")
            return self._fallback_metrics(questions, answers, contexts, ground_truths)

    def _fallback_metrics(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Metrics dự phòng khi RAGAS gặp lỗi (Rate Limit, API Error...)."""
        hits = 0
        mrr_sum = 0.0
        
        for i, (gt, ctx_list) in enumerate(zip(ground_truths, contexts)):
            gt_lower = gt.lower()
            for rank, ctx in enumerate(ctx_list, 1):
                # Kiểm tra overlap từ khóa giữa ground_truth và context
                gt_words = set(gt_lower.split())
                ctx_words = set(ctx.lower().split())
                overlap = len(gt_words & ctx_words) / max(len(gt_words), 1)
                
                if overlap > 0.3:  # Threshold 30% overlap
                    hits += 1
                    mrr_sum += 1.0 / rank
                    break
        
        total = len(questions) if questions else 1
        return {
            "hit_rate": round(hits / total, 4),
            "mrr": round(mrr_sum / total, 4),
            "faithfulness": -1.0,  # Không khả dụng
            "answer_relevancy": -1.0,
            "context_precision": -1.0,
            "context_recall": -1.0,
            "note": "Fallback mode - RAGAS unavailable"
        }

    def compare_baseline_vs_advanced(
        self, 
        collection_name: str,
        dataset_path: str = None
    ) -> Dict[str, Any]:
        """Ablation Study: Đo lường sức mạnh của từng lớp SOTA.
        
        Tách biệt và chạy đo lường từng giai đoạn để thấy % cải thiện cụ thể
        khi lắp thêm tính năng mới vào RAG.
        """
        logger.info("=" * 60)
        logger.info("   ⚔️  ABLATION STUDY - ĐO LƯỜNG TỪNG LỚP SOTA")
        logger.info("=" * 60)
        
        # Vòng 0: Naive RAG (Cốt lõi)
        logger.info("\n🐢 TẦNG 0: NAIVE RAG (Chỉ Search bằng Vector thô)")
        naive_report = self.evaluate_with_ragas(collection_name, dataset_path, mode="naive")
        
        # Vòng 1: Add Hybrid
        logger.info("\n🛠️ TẦNG 1: HYBRID SEARCH (Kết hợp Vector + Keyword BM25)")
        hybrid_report = self.evaluate_with_ragas(collection_name, dataset_path, mode="hybrid_only")
        
        # Vòng 2: Add Reranker
        logger.info("\n🚀 TẦNG 2: RERANKER (Hybrid + Cross-Encoder Reranker)")
        advanced_report = self.evaluate_with_ragas(collection_name, dataset_path, mode="advanced")
        
        # Tổng hợp So sánh
        comparison = {
            "0_naive_baseline": naive_report,
            "1_hybrid_only": hybrid_report,
            "2_advanced_sota": advanced_report,
            "improvements": {}
        }
        
        # Tính delta cải thiện (Từ Naive lên Advanced để đánh giá toàn cục)
        for metric in naive_report["ragas_scores"]:
            naive_val = naive_report["ragas_scores"].get(metric, 0)
            hyb_val = hybrid_report["ragas_scores"].get(metric, 0)
            adv_val = advanced_report["ragas_scores"].get(metric, 0)
            
            if isinstance(naive_val, (int, float)) and isinstance(adv_val, (int, float)):
                if naive_val > 0:
                    comparison["improvements"][metric] = {
                        "naive": naive_val,
                        "hybrid": hyb_val,
                        "advanced": adv_val,
                        "delta_total": round(adv_val - naive_val, 4),
                        "improvement_pct_total": round(((adv_val - naive_val) / max(naive_val, 0.001)) * 100, 2)
                    }
        
        # In bảng báo cáo
        self._print_comparison_report(comparison)
        
        # Xuất file báo cáo
        report_path = os.path.join(os.getcwd(), "tests", "benchmark", "ragas_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        logger.info(f"\n📄 Báo cáo đã lưu tại: {report_path}")
        
        return comparison

    def _print_comparison_report(self, comparison: Dict):
        """In bảng so sánh 3 lớp đẹp mắt ra Terminal."""
        logger.info("\n" + "=" * 80)
        logger.info("   📊 BÁO CÁO ABLATION STUDY: KIỂM CHỨNG TỪNG CẤP ĐỘ SOTA")
        logger.info("=" * 80)
        
        naive = comparison["0_naive_baseline"]["ragas_scores"]
        hybrid = comparison["1_hybrid_only"]["ragas_scores"]
        advanced = comparison["2_advanced_sota"]["ragas_scores"]
        
        logger.info(f"   {'METRIC':<20} | {'Tầng 0: NAIVE':<15} | {'Tầng 1: HYBRID':<15} | {'Tầng 2: SOTA (Rerank)':<20} | Hiệu quả TỐNG%")
        logger.info("-" * 80)
        
        for metric in naive:
            n_val = naive.get(metric, 0)
            h_val = hybrid.get(metric, 0)
            a_val = advanced.get(metric, 0)
            
            if isinstance(n_val, (int, float)) and isinstance(a_val, (int, float)) and n_val >= 0:
                delta = round(a_val - n_val, 4)
                pct = round(((a_val - n_val) / max(n_val, 0.001)) * 100, 1)
                arrow = "🟢 ↑" if delta > 0 else ("🔴 ↓" if delta < 0 else "⚪ →")
                
                logger.info(f"   {metric:<20} | {n_val:<15.4f} | {h_val:<15.4f} | {a_val:<20.4f} | {arrow} +{pct}%")
        
        logger.info("-" * 80)
        # Latency
        n_lat = comparison["0_naive_baseline"]["avg_latency_s"]
        h_lat = comparison["1_hybrid_only"]["avg_latency_s"]
        a_lat = comparison["2_advanced_sota"]["avg_latency_s"]
        logger.info(f"   {'Latency (S)':<20} | {n_lat:<15} | {h_lat:<15} | {a_lat:<20} | (Nhanh -> Chậm)")
        logger.info("=" * 80)
