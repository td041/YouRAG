"""Unit tests for RAGASEvaluator — pipeline runner, fallback metrics, reporting."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_evaluator_deps(mocker):
    """Mock heavy dependencies to prevent real model loading."""
    mocker.patch("src.engine.evaluation.ragas_evaluator.HybridRetriever")
    mocker.patch("src.engine.evaluation.ragas_evaluator.DenseRetriever")
    mocker.patch("src.engine.evaluation.ragas_evaluator.CrossEncoderReranker")
    mocker.patch("src.engine.evaluation.ragas_evaluator.AnswerGenerator")
    mocker.patch("src.engine.evaluation.ragas_evaluator.db_instance")


from src.engine.evaluation.ragas_evaluator import RAGASEvaluator  # noqa: E402


# ── Dataset loading ────────────────────────────────────────────────────────

def test_load_dataset_valid(tmp_path):
    """Kiểm tra load dataset JSON hợp lệ."""
    data = [
        {"question": "Q1", "ground_truth": "A1"},
        {"question": "Q2", "ground_truth": "A2"},
    ]
    fp = tmp_path / "eval.json"
    fp.write_text(json.dumps(data))

    ev = RAGASEvaluator()
    loaded = ev._load_dataset(str(fp))

    assert len(loaded) == 2
    assert loaded[0]["question"] == "Q1"


def test_load_dataset_skips_invalid_items(tmp_path):
    """Kiểm tra bỏ qua item thiếu field bắt buộc."""
    data = [
        {"question": "Q1", "ground_truth": "A1"},
        {"only_question": "Q2"},  # Thiếu ground_truth
    ]
    fp = tmp_path / "eval.json"
    fp.write_text(json.dumps(data))

    ev = RAGASEvaluator()
    loaded = ev._load_dataset(str(fp))

    assert len(loaded) == 1


def test_load_dataset_raises_when_no_valid_items(tmp_path):
    """Kiểm tra raise ValueError khi không có item hợp lệ."""
    data = [{"bad": "data"}]
    fp = tmp_path / "eval.json"
    fp.write_text(json.dumps(data))

    ev = RAGASEvaluator()
    with pytest.raises(ValueError, match="không có item hợp lệ"):
        ev._load_dataset(str(fp))


# ── Pipeline runner ────────────────────────────────────────────────────────

def test_run_pipeline_naive_mode():
    """Kiểm tra pipeline mode 'naive' chỉ dùng DenseRetriever."""
    ev = RAGASEvaluator()
    ev.naive_retriever = MagicMock()
    ev.naive_retriever.search.return_value = [{"content": "result"}]
    ev.generator = MagicMock()
    ev.generator.generate.return_value = "Answer"

    result = ev._run_pipeline("query", "col", "naive")

    assert result["answer"] == "Answer"
    assert result["contexts"] == ["result"]
    ev.naive_retriever.search.assert_called_once()


def test_run_pipeline_hybrid_mode():
    """Kiểm tra pipeline mode 'hybrid' dùng HybridRetriever."""
    ev = RAGASEvaluator()
    ev.hybrid_retriever = MagicMock()
    ev.hybrid_retriever.search.return_value = [{"content": "hybrid_result"}]
    ev.generator = MagicMock()
    ev.generator.generate.return_value = "Hybrid answer"

    result = ev._run_pipeline("query", "col", "hybrid")

    assert result["answer"] == "Hybrid answer"
    ev.hybrid_retriever.search.assert_called_once()


def test_run_pipeline_advanced_mode():
    """Kiểm tra pipeline mode 'advanced' dùng Hybrid + Reranker."""
    ev = RAGASEvaluator()
    ev.hybrid_retriever = MagicMock()
    ev.hybrid_retriever.search.return_value = [{"content": "raw"}, {"content": "raw2"}]
    ev.reranker = MagicMock()
    ev.reranker.rerank.return_value = [{"content": "reranked"}]
    ev.generator = MagicMock()
    ev.generator.generate.return_value = "Advanced answer"

    result = ev._run_pipeline("query", "col", "advanced")

    assert result["answer"] == "Advanced answer"
    ev.reranker.rerank.assert_called_once()


# ── Fallback metrics ──────────────────────────────────────────────────────

def test_fallback_metrics_hit():
    """Kiểm tra fallback metrics phát hiện hit khi overlap > 25%."""
    ev = RAGASEvaluator()
    result = ev._fallback_metrics(
        questions=["What is AI?"],
        answers=["AI is intelligence"],
        contexts=[["AI is artificial intelligence used in many fields"]],
        ground_truths=["AI is artificial intelligence"],
    )

    assert result["hit_rate"] > 0
    assert result["_fallback"] is True
    assert result["faithfulness"] is None


def test_fallback_metrics_no_hit():
    """Kiểm tra fallback metrics khi không có overlap."""
    ev = RAGASEvaluator()
    result = ev._fallback_metrics(
        questions=["xyz?"],
        answers=["abc"],
        contexts=[["completely different content about cats"]],
        ground_truths=["something about quantum physics and dark matter"],
    )

    assert result["hit_rate"] == 0.0
    assert result["mrr"] == 0.0


# ── compute_ragas_metrics fallback ────────────────────────────────────────

def test_compute_ragas_metrics_falls_back_on_import_error():
    """Kiểm tra fallback khi RAGAS library không khả dụng."""
    ev = RAGASEvaluator()

    with patch.dict("sys.modules", {"ragas": None}):
        result = ev._compute_ragas_metrics(
            questions=["Q"],
            answers=["A"],
            contexts_list=[["ctx"]],
            ground_truths=["gt"],
        )

    # Should fallback to keyword overlap metrics
    assert "_fallback" in result or "hit_rate" in result or result.get("faithfulness") is None


# ── evaluate (single mode) ────────────────────────────────────────────────

def test_evaluate_single_mode(tmp_path):
    """Kiểm tra evaluate chạy 1 mode và trả về báo cáo đầy đủ."""
    data = [{"question": "Q1", "ground_truth": "A1"}]
    fp = tmp_path / "eval.json"
    fp.write_text(json.dumps(data))

    ev = RAGASEvaluator()
    ev._run_pipeline = MagicMock(return_value={
        "answer": "ans", "contexts": ["ctx"], "latency": 0.5, "num_chunks": 1
    })
    ev._compute_ragas_metrics = MagicMock(return_value={
        "faithfulness": 0.9, "_per_question": [{"faithfulness": 0.9}]
    })

    result = ev.evaluate("col", mode="naive", dataset_path=str(fp))

    assert result["mode"] == "naive"
    assert result["total_questions"] == 1
    assert result["ragas_scores"]["faithfulness"] == 0.9


# ── Markdown report builder ──────────────────────────────────────────────

def test_build_markdown_report():
    """Kiểm tra Markdown report chứa heading + table."""
    ev = RAGASEvaluator()
    comparison = {
        "collection": "test_col",
        "generated_at": "2026-01-01",
        "0_naive": {"total_questions": 5, "ragas_scores": {"faithfulness": 0.8}, "avg_latency_s": 1.0},
        "1_hybrid": {"ragas_scores": {"faithfulness": 0.85}, "avg_latency_s": 1.2},
        "2_advanced": {"ragas_scores": {"faithfulness": 0.95}, "avg_latency_s": 1.5},
        "improvements": {
            "faithfulness": {
                "naive": 0.8, "hybrid": 0.85, "advanced": 0.95,
                "delta_hybrid_vs_naive": 0.05, "delta_advanced_vs_hybrid": 0.1,
                "delta_total": 0.15, "improvement_pct": 18.75
            }
        },
    }

    md = ev._build_markdown_report(comparison)

    assert "# YouRAG" in md
    assert "| faithfulness" in md
    assert "Ablation Study" in md


# ── save_reports ─────────────────────────────────────────────────────────

def test_save_reports(tmp_path, mocker):
    """Kiểm tra lưu JSON + Markdown report ra file."""
    mocker.patch("src.engine.evaluation.ragas_evaluator._BENCHMARK_DIR", str(tmp_path))
    mocker.patch("src.engine.evaluation.ragas_evaluator._DEFAULT_REPORT_JSON", str(tmp_path / "report.json"))
    mocker.patch("src.engine.evaluation.ragas_evaluator._DEFAULT_REPORT_MD", str(tmp_path / "report.md"))

    ev = RAGASEvaluator()
    ev._build_markdown_report = MagicMock(return_value="# Report")

    comparison = {"collection": "test", "data": "value"}
    ev._save_reports(comparison)

    assert os.path.exists(tmp_path / "report.json")
    assert os.path.exists(tmp_path / "report.md")


# ── Legacy alias ─────────────────────────────────────────────────────────

def test_evaluate_with_ragas_alias():
    """Kiểm tra backward-compatible alias gọi evaluate đúng."""
    ev = RAGASEvaluator()
    ev.evaluate = MagicMock(return_value={"mode": "advanced"})

    ev.evaluate_with_ragas("col", mode="hybrid")

    ev.evaluate.assert_called_once_with("col", "hybrid", None)
