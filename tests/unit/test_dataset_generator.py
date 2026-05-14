"""Unit tests for EvalDatasetGenerator — tạo bộ đánh giá RAGAS."""

import json
import os
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_deps(mocker):
    """Mock LLMClient and db_instance."""
    mock_llm_cls = mocker.patch("src.engine.evaluation.dataset_generator.LLMClient")
    mock_llm = MagicMock()
    mock_llm.chat_complete.return_value = '{"question": "Nội dung gì?", "ground_truth": "Nói về AI"}'
    mock_llm_cls.return_value = mock_llm

    mock_db = MagicMock()
    mocker.patch("src.engine.evaluation.dataset_generator.db_instance", mock_db)

    return mock_llm, mock_db


from src.engine.evaluation.dataset_generator import EvalDatasetGenerator  # noqa: E402


# ── _fetch_chunks ──────────────────────────────────────────────────────────

def test_fetch_chunks_parses_records(mock_deps):
    """Kiểm tra kéo chunks từ Qdrant và parse payload đúng."""
    _, mock_db = mock_deps

    rec = MagicMock()
    rec.id = "id_1"
    rec.payload = {"text": "Hello world", "chunk_index": 0, "start_time": 1.0, "end_time": 5.0}
    mock_db.client.scroll.return_value = ([rec], None)

    gen = EvalDatasetGenerator(num_questions=5)
    chunks = gen._fetch_chunks("test_col")

    assert len(chunks) == 1
    assert chunks[0]["content"] == "Hello world"
    assert chunks[0]["index"] == 0


def test_fetch_chunks_strips_context_enrichment(mock_deps):
    """Kiểm tra bóc tách context enrichment prefix."""
    _, mock_db = mock_deps

    rec = MagicMock()
    rec.id = "id_2"
    rec.payload = {"text": "Context prefix\n\nActual content", "chunk_index": 1, "start_time": 0, "end_time": 0}
    mock_db.client.scroll.return_value = ([rec], None)

    gen = EvalDatasetGenerator()
    chunks = gen._fetch_chunks("col")

    assert chunks[0]["content"] == "Actual content"


def test_fetch_chunks_pagination(mock_deps):
    """Kiểm tra pagination: scroll nhiều lần cho đến next_offset=None."""
    _, mock_db = mock_deps

    rec1 = MagicMock()
    rec1.id = "1"
    rec1.payload = {"text": "A", "chunk_index": 0, "start_time": 0, "end_time": 0}

    rec2 = MagicMock()
    rec2.id = "2"
    rec2.payload = {"text": "B", "chunk_index": 1, "start_time": 0, "end_time": 0}

    # Lần 1: trả về rec1 + next_offset = "abc"; Lần 2: trả về rec2 + None
    mock_db.client.scroll.side_effect = [([rec1], "abc"), ([rec2], None)]

    gen = EvalDatasetGenerator()
    chunks = gen._fetch_chunks("col")

    assert len(chunks) == 2
    assert mock_db.client.scroll.call_count == 2


# ── _sample_chunks ─────────────────────────────────────────────────────────

def test_sample_chunks_returns_all_when_fewer_than_num():
    """Kiểm tra trả về toàn bộ nếu số chunks <= num_questions."""
    gen = EvalDatasetGenerator(num_questions=10)
    chunks = [{"content": f"C{i}"} for i in range(5)]
    assert gen._sample_chunks(chunks) == chunks


def test_sample_chunks_uniform_sampling():
    """Kiểm tra trích mẫu đều khi có nhiều chunks hơn num_questions."""
    gen = EvalDatasetGenerator(num_questions=3)
    chunks = [{"content": f"C{i}"} for i in range(20)]
    sampled = gen._sample_chunks(chunks)
    assert len(sampled) == 3


# ── _generate_qa_from_chunk ────────────────────────────────────────────────

def test_generate_qa_success(mock_deps):
    """Kiểm tra sinh QA thành công từ chunk."""
    mock_llm, _ = mock_deps

    gen = EvalDatasetGenerator()
    chunk = {"content": "AI rất hay", "start_time": 0, "end_time": 5, "index": 0}
    qa = gen._generate_qa_from_chunk(chunk, 0)

    assert qa["question"] == "Nội dung gì?"
    assert qa["ground_truth"] == "Nói về AI"
    assert qa["reference_context"] == "AI rất hay"


def test_generate_qa_fallback_on_error(mock_deps):
    """Kiểm tra fallback khi LLM trả JSON lỗi."""
    mock_llm, _ = mock_deps
    mock_llm.chat_complete.return_value = "not valid json"

    gen = EvalDatasetGenerator()
    chunk = {"content": "AI content", "start_time": 0, "end_time": 5, "index": 0}
    qa = gen._generate_qa_from_chunk(chunk, 0)

    # Fallback: tạo câu hỏi mặc định
    assert "mốc" in qa["question"]
    assert qa["reference_context"] == "AI content"


def test_generate_qa_strips_markdown_wrapper(mock_deps):
    """Kiểm tra bóc markdown wrapper (```json...```) từ LLM response."""
    mock_llm, _ = mock_deps
    mock_llm.chat_complete.return_value = '```json\n{"question": "Q1", "ground_truth": "A1"}\n```'

    gen = EvalDatasetGenerator()
    chunk = {"content": "test", "start_time": 0, "end_time": 1, "index": 0}
    qa = gen._generate_qa_from_chunk(chunk, 0)

    assert qa["question"] == "Q1"


# ── generate (Pipeline chính) ─────────────────────────────────────────────

def test_generate_full_pipeline(mock_deps, tmp_path):
    """Kiểm tra pipeline: fetch → sample → generate QA → lưu file JSON."""
    _, mock_db = mock_deps

    # 2 chunks
    recs = []
    for i in range(2):
        r = MagicMock()
        r.id = f"id_{i}"
        r.payload = {"text": f"Content {i}", "chunk_index": i, "start_time": float(i), "end_time": float(i + 1)}
        recs.append(r)
    mock_db.client.scroll.return_value = (recs, None)

    output = str(tmp_path / "eval.json")
    gen = EvalDatasetGenerator(num_questions=2)
    dataset = gen.generate("test_col", output_path=output)

    assert len(dataset) == 2
    assert os.path.exists(output)

    with open(output) as f:
        saved = json.load(f)
    assert len(saved) == 2


def test_generate_returns_empty_on_empty_collection(mock_deps):
    """Kiểm tra trả về [] khi collection rỗng."""
    _, mock_db = mock_deps
    mock_db.client.scroll.return_value = ([], None)

    gen = EvalDatasetGenerator()
    result = gen.generate("empty_col")

    assert result == []
