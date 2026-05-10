"""
RAGAS Evaluator - Đánh giá RAG Pipeline theo chuẩn nghiên cứu SOTA.

Tích hợp RAGAS 0.4.x để đo lường toàn diện chất lượng YouRAG với 5 metrics:
- Faithfulness         : Câu trả lời trung thành với context không? (Hallucination check)
- Answer Relevancy     : Câu trả lời có liên quan đến câu hỏi không?
- Context Precision    : Context truy xuất được có chính xác, không nhiễu không?
- Context Recall       : Context có bao phủ đủ ground truth không?
- Factual Correctness  : Câu trả lời đúng thực tế so với ground truth không?

Ablation Study 3 tầng:
  Tầng 0 - NAIVE      : Dense Search only (baseline)
  Tầng 1 - HYBRID     : Dense + BM25 Sparse (RRF fusion)
  Tầng 2 - ADVANCED   : Hybrid + Cross-Encoder Reranker (SOTA)
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.core.config import settings
from src.core.database import db_instance
from src.core.logger import logger
from src.engine.generation.answer_generator import AnswerGenerator
from src.engine.ranking.cross_encoder import CrossEncoderReranker
from src.engine.retrieval.dense_search import DenseRetriever
from src.engine.retrieval.hybrid_search import HybridRetriever

# Đường dẫn mặc định cho dataset và report
_BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "../../../tests/benchmark")
_DEFAULT_DATASET = os.path.join(_BENCHMARK_DIR, "eval_dataset.json")
_DEFAULT_REPORT_JSON = os.path.join(_BENCHMARK_DIR, "ragas_report.json")
_DEFAULT_REPORT_MD = os.path.join(_BENCHMARK_DIR, "ragas_report.md")


class RAGASEvaluator:
    """Đánh giá End-to-End hệ thống YouRAG theo chuẩn RAGAS 0.4.x.

    Pipeline:
    1. Load eval_dataset.json (question + ground_truth + reference_context)
    2. Chạy RAG pipeline 3 chế độ: naive / hybrid / advanced
    3. Thu thập (user_input, response, retrieved_contexts, reference)
    4. Tính 5 RAGAS metrics với Groq LLM + HuggingFace embeddings
    5. Xuất báo cáo JSON + Markdown chi tiết
    """

    def __init__(self) -> None:
        self.hybrid_retriever = HybridRetriever(top_k=10)
        self.naive_retriever = DenseRetriever(top_k=5)
        self.reranker = CrossEncoderReranker()
        self.generator = AnswerGenerator()
        self.db = db_instance

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load eval dataset từ JSON. Hỗ trợ cả format cũ lẫn format mới."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        validated = []
        for item in data:
            if "question" not in item or "ground_truth" not in item:
                logger.warning(f"⚠️  Bỏ qua item thiếu field: {list(item.keys())}")
                continue
            validated.append(item)

        if not validated:
            raise ValueError(
                f"Dataset tại '{dataset_path}' không có item hợp lệ. "
                "Mỗi item phải có 'question' và 'ground_truth'. "
                "Hãy chạy --generate trước."
            )
        return validated

    # ------------------------------------------------------------------
    # RAG Pipeline runner
    # ------------------------------------------------------------------

    def _run_pipeline(
        self, query: str, collection_name: str, mode: str
    ) -> Dict[str, Any]:
        """Chạy một trong 3 chế độ RAG, trả về answer + contexts + latency."""
        t0 = time.time()

        if mode == "naive":
            results = self.naive_retriever.search(query, collection_name)
            top_results = results[:5]

        elif mode == "hybrid":
            candidates = self.hybrid_retriever.search(query, collection_name)
            top_results = candidates[:5]

        else:  # advanced
            candidates = self.hybrid_retriever.search(query, collection_name)
            top_results = self.reranker.rerank(query=query, chunks=candidates, top_k=5)

        contexts = [r["content"] for r in top_results if r.get("content")]
        answer = self.generator.generate(query=query, retrieved_chunks=top_results)

        return {
            "answer": answer,
            "contexts": contexts,
            "latency": round(time.time() - t0, 3),
            "num_chunks": len(top_results),
        }

    # ------------------------------------------------------------------
    # RAGAS metrics computation
    # ------------------------------------------------------------------

    def _build_ragas_llm(self) -> Any:
        """Tạo Groq-backed LLM cho RAGAS thông qua OpenAI-compatible API."""
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        groq_key = (
            settings.GROQ_API_KEY.get_secret_value()
            if settings.GROQ_API_KEY
            else os.getenv("GROQ_API_KEY", "")
        )
        llm = ChatOpenAI(
            model="llama-3.1-8b-instant",
            openai_api_key=groq_key,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.0,
            max_tokens=1024,
            n=1,
        )
        return LangchainLLMWrapper(llm)

    def _build_ragas_embeddings(self) -> Any:
        """Tạo HuggingFace embeddings cho RAGAS (dùng lại sentence-transformers đã cài)."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        hf_emb = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return LangchainEmbeddingsWrapper(hf_emb)

    def _compute_ragas_metrics(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str],
    ) -> Dict[str, Any]:
        """Tính 5 RAGAS metrics chuẩn SOTA với RAGAS 0.4.x API."""
        try:
            from ragas import evaluate
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
            from ragas.metrics import (
                _answer_relevancy,
                _context_precision,
                _context_recall,
                _faithfulness,
            )
            from ragas.metrics._factual_correctness import FactualCorrectness

            ragas_llm = self._build_ragas_llm()
            ragas_emb = self._build_ragas_embeddings()

            # Cấu hình LLM + Embeddings cho từng metric
            _faithfulness.llm = ragas_llm
            _answer_relevancy.llm = ragas_llm
            _answer_relevancy.embeddings = ragas_emb
            _context_precision.llm = ragas_llm
            _context_recall.llm = ragas_llm

            factual_correctness = FactualCorrectness()
            factual_correctness.llm = ragas_llm

            # Build EvaluationDataset theo RAGAS 0.4.x schema
            samples = [
                SingleTurnSample(
                    user_input=q,
                    response=a,
                    retrieved_contexts=ctx,
                    reference=gt,
                )
                for q, a, ctx, gt in zip(questions, answers, contexts_list, ground_truths)
            ]
            dataset = EvaluationDataset(samples=samples)

            logger.info("🔬 Đang tính RAGAS metrics (có thể mất 1-3 phút)...")
            result = evaluate(
                dataset=dataset,
                metrics=[
                    _faithfulness,
                    _answer_relevancy,
                    _context_precision,
                    _context_recall,
                    factual_correctness,
                ],
                raise_exceptions=False,
                show_progress=True,
            )

            # Lấy điểm trung bình
            scores: Dict[str, Any] = {}
            df = result.to_pandas()

            metric_cols = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "factual_correctness",
            ]
            for col in metric_cols:
                if col in df.columns:
                    scores[col] = round(float(df[col].mean(skipna=True)), 4)

            # Per-question chi tiết
            scores["_per_question"] = df[
                [c for c in metric_cols if c in df.columns]
            ].to_dict(orient="records")

            logger.info(f"📊 Scores: { {k: v for k, v in scores.items() if not k.startswith('_')} }")
            return scores

        except Exception as e:
            logger.error(f"❌ Lỗi RAGAS: {e}")
            logger.warning("⚠️  Fallback: tính Hit Rate + MRR thủ công")
            return self._fallback_metrics(questions, answers, contexts_list, ground_truths)

    def _fallback_metrics(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """Metrics dự phòng khi RAGAS gặp lỗi (dựa trên keyword overlap)."""
        hits, mrr_sum = 0, 0.0

        for gt, ctx_list in zip(ground_truths, contexts):
            gt_words = set(gt.lower().split())
            for rank, ctx in enumerate(ctx_list, 1):
                overlap = len(gt_words & set(ctx.lower().split())) / max(len(gt_words), 1)
                if overlap > 0.25:
                    hits += 1
                    mrr_sum += 1.0 / rank
                    break

        total = max(len(questions), 1)
        return {
            "hit_rate": round(hits / total, 4),
            "mrr": round(mrr_sum / total, 4),
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "factual_correctness": None,
            "_fallback": True,
            "_note": "RAGAS unavailable — showing keyword-overlap proxies only",
        }

    # ------------------------------------------------------------------
    # Single-mode evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        collection_name: str,
        mode: str = "advanced",
        dataset_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Đánh giá 1 chế độ RAG và trả về báo cáo đầy đủ.

        Args:
            collection_name: Tên collection Qdrant
            mode: 'naive' | 'hybrid' | 'advanced'
            dataset_path: Đường dẫn eval_dataset.json (mặc định: tests/benchmark/)
        """
        path = dataset_path or _DEFAULT_DATASET
        dataset = self._load_dataset(path)

        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 RAGAS [{mode.upper()}]: {len(dataset)} câu hỏi → [{collection_name}]")
        logger.info(f"{'='*60}")

        questions, answers, contexts_list, ground_truths = [], [], [], []
        per_question: List[Dict] = []
        latencies: List[float] = []

        for idx, item in enumerate(dataset, 1):
            query = item["question"]
            gt = item["ground_truth"]
            logger.info(f"\n[{idx}/{len(dataset)}] Q: {query[:80]}")

            out = self._run_pipeline(query, collection_name, mode)

            questions.append(query)
            answers.append(out["answer"])
            contexts_list.append(out["contexts"])
            ground_truths.append(gt)
            latencies.append(out["latency"])

            per_question.append(
                {
                    "question": query,
                    "answer": out["answer"],
                    "ground_truth": gt,
                    "num_contexts": out["num_chunks"],
                    "latency_s": out["latency"],
                }
            )
            logger.info(f"   → {out['answer'][:100]}... ({out['latency']}s)")

        ragas_scores = self._compute_ragas_metrics(
            questions, answers, contexts_list, ground_truths
        )

        # Gắn per-question RAGAS scores vào per_question list
        per_q_ragas = ragas_scores.pop("_per_question", [])
        for i, row in enumerate(per_q_ragas):
            if i < len(per_question):
                per_question[i]["ragas_scores"] = row

        avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

        return {
            "mode": mode,
            "collection": collection_name,
            "total_questions": len(dataset),
            "avg_latency_s": avg_lat,
            "ragas_scores": ragas_scores,
            "per_question": per_question,
        }

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------

    def compare_baseline_vs_advanced(
        self,
        collection_name: str,
        dataset_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ablation Study 3 tầng: Naive → Hybrid → Advanced.

        Đo lường mức độ cải thiện khi bổ sung từng lớp SOTA vào pipeline.
        """
        logger.info("\n" + "=" * 70)
        logger.info("   ⚔️  ABLATION STUDY — ĐO LƯỜNG TỪNG LỚP SOTA")
        logger.info("=" * 70)

        logger.info("\n🐢 TẦNG 0: NAIVE RAG (Dense Search only)")
        naive = self.evaluate(collection_name, "naive", dataset_path)

        logger.info("\n🛠️  TẦNG 1: HYBRID SEARCH (Dense + BM25 RRF)")
        hybrid = self.evaluate(collection_name, "hybrid", dataset_path)

        logger.info("\n🚀 TẦNG 2: ADVANCED SOTA (Hybrid + Cross-Encoder Reranker)")
        advanced = self.evaluate(collection_name, "advanced", dataset_path)

        # Tính delta giữa các tầng
        improvements: Dict[str, Any] = {}
        core_metrics = [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "factual_correctness",
        ]
        for metric in core_metrics:
            n = naive["ragas_scores"].get(metric)
            h = hybrid["ragas_scores"].get(metric)
            a = advanced["ragas_scores"].get(metric)
            if n is None or a is None:
                continue
            delta_total = round(a - n, 4)
            delta_hybrid = round(h - n, 4) if h is not None else None
            delta_rerank = round(a - h, 4) if h is not None else None
            improvements[metric] = {
                "naive": n,
                "hybrid": h,
                "advanced": a,
                "delta_hybrid_vs_naive": delta_hybrid,
                "delta_advanced_vs_hybrid": delta_rerank,
                "delta_total": delta_total,
                "improvement_pct": round((delta_total / max(abs(n), 0.001)) * 100, 2),
            }

        comparison = {
            "generated_at": datetime.now().isoformat(),
            "collection": collection_name,
            "0_naive": naive,
            "1_hybrid": hybrid,
            "2_advanced": advanced,
            "improvements": improvements,
        }

        self._print_report(comparison)
        self._save_reports(comparison)

        return comparison

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_report(self, comparison: Dict) -> None:
        """In bảng so sánh 3 tầng ra terminal."""
        naive_s = comparison["0_naive"]["ragas_scores"]
        hybrid_s = comparison["1_hybrid"]["ragas_scores"]
        adv_s = comparison["2_advanced"]["ragas_scores"]

        col_w = 18
        logger.info("\n" + "=" * 90)
        logger.info("   📊 BÁO CÁO ABLATION STUDY — YouRAG RAGAS Evaluation")
        logger.info("=" * 90)
        header = (
            f"   {'METRIC':<{col_w}} | {'NAIVE (Tầng 0)':<16} | "
            f"{'HYBRID (Tầng 1)':<16} | {'ADVANCED (Tầng 2)':<18} | Δ Total"
        )
        logger.info(header)
        logger.info("-" * 90)

        core_metrics = [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "factual_correctness",
        ]
        for metric in core_metrics:
            n = naive_s.get(metric)
            h = hybrid_s.get(metric)
            a = adv_s.get(metric)
            if n is None:
                logger.info(f"   {metric:<{col_w}} | {'N/A':<16} | {'N/A':<16} | {'N/A':<18} | —")
                continue

            delta = round(a - n, 4) if a is not None else 0.0
            pct = round((delta / max(abs(n), 0.001)) * 100, 1)
            arrow = "🟢 ↑" if delta > 0.001 else ("🔴 ↓" if delta < -0.001 else "⚪ →")
            sign = "+" if delta >= 0 else ""
            n_str = f"{n:.4f}" if n is not None else "N/A"
            h_str = f"{h:.4f}" if h is not None else "N/A"
            a_str = f"{a:.4f}" if a is not None else "N/A"

            logger.info(
                f"   {metric:<{col_w}} | {n_str:<16} | {h_str:<16} | "
                f"{a_str:<18} | {arrow} {sign}{pct}%"
            )

        logger.info("-" * 90)
        n_lat = comparison["0_naive"]["avg_latency_s"]
        h_lat = comparison["1_hybrid"]["avg_latency_s"]
        a_lat = comparison["2_advanced"]["avg_latency_s"]
        logger.info(
            f"   {'Latency (s)':<{col_w}} | {n_lat:<16} | {h_lat:<16} | {a_lat:<18} |"
        )
        logger.info("=" * 90)

    def _save_reports(self, comparison: Dict) -> None:
        """Lưu báo cáo dạng JSON + Markdown."""
        os.makedirs(_BENCHMARK_DIR, exist_ok=True)

        # JSON
        with open(_DEFAULT_REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"📄 JSON report: {_DEFAULT_REPORT_JSON}")

        # Markdown
        md = self._build_markdown_report(comparison)
        with open(_DEFAULT_REPORT_MD, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"📝 Markdown report: {_DEFAULT_REPORT_MD}")

    def _build_markdown_report(self, comparison: Dict) -> str:
        """Tạo Markdown report cho GitHub / Notion."""
        lines = [
            "# YouRAG — RAGAS Evaluation Report",
            "",
            f"**Collection:** `{comparison['collection']}`  ",
            f"**Generated:** {comparison['generated_at']}  ",
            f"**Questions:** {comparison['0_naive']['total_questions']}",
            "",
            "## Ablation Study Results",
            "",
            "| Metric | Naive (Tầng 0) | Hybrid (Tầng 1) | Advanced (Tầng 2) | Δ Total | Improvement |",
            "|--------|---------------|-----------------|-------------------|---------|-------------|",
        ]

        core_metrics = [
            "faithfulness", "answer_relevancy", "context_precision",
            "context_recall", "factual_correctness",
        ]
        improvements = comparison.get("improvements", {})
        naive_s = comparison["0_naive"]["ragas_scores"]

        for metric in core_metrics:
            if metric in improvements:
                imp = improvements[metric]
                n = f"{imp['naive']:.4f}"
                h = f"{imp['hybrid']:.4f}" if imp["hybrid"] is not None else "N/A"
                a = f"{imp['advanced']:.4f}" if imp["advanced"] is not None else "N/A"
                delta = imp["delta_total"]
                pct = imp["improvement_pct"]
                sign = "+" if delta >= 0 else ""
                icon = "🟢" if delta > 0.001 else ("🔴" if delta < -0.001 else "⚪")
                lines.append(
                    f"| {metric} | {n} | {h} | {a} | {sign}{delta:.4f} | {icon} {sign}{pct}% |"
                )
            else:
                n = naive_s.get(metric)
                lines.append(
                    f"| {metric} | {'N/A' if n is None else f'{n:.4f}'} | N/A | N/A | — | — |"
                )

        # Latency
        n_lat = comparison["0_naive"]["avg_latency_s"]
        h_lat = comparison["1_hybrid"]["avg_latency_s"]
        a_lat = comparison["2_advanced"]["avg_latency_s"]
        lines += [
            f"| **Latency (s)** | {n_lat} | {h_lat} | {a_lat} | — | — |",
            "",
            "## Metric Definitions",
            "",
            "| Metric | Mô tả |",
            "|--------|-------|",
            "| **Faithfulness** | Câu trả lời trung thành với context, không hallucinate |",
            "| **Answer Relevancy** | Câu trả lời có liên quan đến câu hỏi không |",
            "| **Context Precision** | Context truy xuất có chính xác, ít nhiễu không |",
            "| **Context Recall** | Context có bao phủ đủ ground truth không |",
            "| **Factual Correctness** | Câu trả lời đúng thực tế so với ground truth |",
            "",
            "---",
            "*Evaluated with [RAGAS 0.4.x](https://docs.ragas.io) + Groq llama-3.1-8b-instant*",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Legacy alias (backward compat với run_benchmark.py)
    # ------------------------------------------------------------------

    def evaluate_with_ragas(
        self,
        collection_name: str,
        dataset_path: Optional[str] = None,
        mode: str = "advanced",
    ) -> Dict[str, Any]:
        """Alias giữ backward compatibility với run_benchmark.py."""
        return self.evaluate(collection_name, mode, dataset_path)
