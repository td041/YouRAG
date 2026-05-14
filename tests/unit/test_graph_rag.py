"""Unit tests for GraphRetriever and KnowledgeGraphBuilder."""

import json
import os
import pickle
import pytest
from unittest.mock import MagicMock

import networkx as nx


@pytest.fixture(autouse=True)
def mock_graph_deps(mocker):
    """Mock LLMClient and db_instance."""
    mock_llm_cls = mocker.patch("src.engine.retrieval.graph_rag.LLMClient")
    mock_llm = MagicMock()
    mock_llm.chat_complete.return_value = '[]'
    mock_llm_cls.return_value = mock_llm

    mock_db = MagicMock()
    mocker.patch("src.engine.retrieval.graph_rag.db_instance", mock_db)

    return mock_llm, mock_db


from src.engine.retrieval.graph_rag import KnowledgeGraphBuilder, GraphRetriever  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# KnowledgeGraphBuilder tests
# ═══════════════════════════════════════════════════════════════════════════

def test_extract_triples_success(mock_graph_deps):
    """Kiểm tra trích xuất triples thành công từ LLM response."""
    mock_llm, _ = mock_graph_deps
    mock_llm.chat_complete.return_value = json.dumps([
        {"subject": "AI", "predicate": "uses", "object": "Neural Networks"}
    ])

    builder = KnowledgeGraphBuilder()
    triples = builder._extract_triples("AI uses Neural Networks for learning.", 0)

    assert len(triples) == 1
    assert triples[0]["subject"] == "AI"
    assert triples[0]["chunk_index"] == 0


def test_extract_triples_error_returns_empty(mock_graph_deps):
    """Kiểm tra trả về [] khi LLM lỗi."""
    mock_llm, _ = mock_graph_deps
    mock_llm.chat_complete.side_effect = Exception("API error")

    builder = KnowledgeGraphBuilder()
    triples = builder._extract_triples("text", 0)

    assert triples == []


def test_extract_triples_strips_markdown(mock_graph_deps):
    """Kiểm tra bóc markdown wrapper từ response."""
    mock_llm, _ = mock_graph_deps
    mock_llm.chat_complete.return_value = '```json\n[{"subject":"A","predicate":"r","object":"B"}]\n```'

    builder = KnowledgeGraphBuilder()
    triples = builder._extract_triples("text", 0)

    assert len(triples) == 1
    assert triples[0]["subject"] == "A"


def test_build_graph_empty_collection(mock_graph_deps):
    """Kiểm tra build graph trả về graph rỗng khi collection rỗng."""
    _, mock_db = mock_graph_deps
    mock_db.client.scroll.return_value = ([], None)

    builder = KnowledgeGraphBuilder()
    G = builder.build_graph("empty_col")

    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_build_graph_creates_nodes_and_edges(mock_graph_deps, tmp_path, mocker):
    """Kiểm tra build graph tạo nodes và edges đúng."""
    mock_llm, mock_db = mock_graph_deps

    # Mock Qdrant records
    rec = MagicMock()
    rec.payload = {"text": "Hello", "chunk_index": 0, "start_time": 0}
    mock_db.client.scroll.return_value = ([rec], None)

    # Mock LLM trả về triples
    mock_llm.chat_complete.return_value = json.dumps([
        {"subject": "Python", "predicate": "is", "object": "Programming Language"}
    ])

    # Override GRAPH_DIR to temp
    mocker.patch.object(KnowledgeGraphBuilder, "GRAPH_DIR", str(tmp_path))

    builder = KnowledgeGraphBuilder()
    G = builder.build_graph("test_col")

    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    assert G.has_node("Python")
    assert G.has_node("Programming Language")

    # Verify pickle file was saved
    assert os.path.exists(tmp_path / "test_col.gpickle")


# ═══════════════════════════════════════════════════════════════════════════
# GraphRetriever tests
# ═══════════════════════════════════════════════════════════════════════════

def _build_sample_graph():
    """Helper: Tạo Graph mẫu cho tests."""
    G = nx.DiGraph()
    G.add_node("Python", chunk_indices=[0, 1], entity_type="entity")
    G.add_node("AI", chunk_indices=[1, 2], entity_type="entity")
    G.add_node("Machine Learning", chunk_indices=[2], entity_type="entity")
    G.add_edge("AI", "Machine Learning", relation="sử dụng", chunk_index=1)
    G.add_edge("Python", "AI", relation="lập trình cho", chunk_index=0)
    return G


def test_load_graph_from_disk(mock_graph_deps, tmp_path, mocker):
    """Kiểm tra load graph từ pickle file."""
    mocker.patch.object(GraphRetriever, "GRAPH_DIR", str(tmp_path))

    G = _build_sample_graph()
    with open(tmp_path / "test_col.gpickle", "wb") as f:
        pickle.dump(G, f)

    retriever = GraphRetriever()
    loaded = retriever._load_graph("test_col")

    assert loaded is not None
    assert loaded.number_of_nodes() == 3


def test_load_graph_returns_none_when_missing(mock_graph_deps, tmp_path, mocker):
    """Kiểm tra trả về None khi file graph chưa tồn tại."""
    mocker.patch.object(GraphRetriever, "GRAPH_DIR", str(tmp_path))

    retriever = GraphRetriever()
    result = retriever._load_graph("nonexistent")

    assert result is None


def test_load_graph_caches(mock_graph_deps, tmp_path, mocker):
    """Kiểm tra graph được cache sau lần load đầu tiên."""
    mocker.patch.object(GraphRetriever, "GRAPH_DIR", str(tmp_path))

    G = _build_sample_graph()
    with open(tmp_path / "cached_col.gpickle", "wb") as f:
        pickle.dump(G, f)

    retriever = GraphRetriever()
    g1 = retriever._load_graph("cached_col")
    g2 = retriever._load_graph("cached_col")

    assert g1 is g2  # Cùng 1 object (cached)


def test_extract_query_entities(mock_graph_deps):
    """Kiểm tra trích entity từ query."""
    mock_llm, _ = mock_graph_deps
    mock_llm.chat_complete.return_value = '["Python", "AI"]'

    retriever = GraphRetriever()
    entities = retriever._extract_query_entities("Python dùng cho AI?")

    assert "Python" in entities
    assert "AI" in entities


def test_extract_query_entities_fallback(mock_graph_deps):
    """Kiểm tra fallback khi LLM lỗi."""
    mock_llm, _ = mock_graph_deps
    mock_llm.chat_complete.side_effect = Exception("LLM down")

    retriever = GraphRetriever()
    entities = retriever._extract_query_entities("Python AI query")

    # Fallback: regex tìm từ viết hoa hoặc tách query
    assert len(entities) > 0


def test_find_matching_nodes():
    """Kiểm tra tìm node khớp với fuzzy matching."""
    G = _build_sample_graph()
    retriever = GraphRetriever()

    matched = retriever._find_matching_nodes(G, ["Python"])
    assert "Python" in matched

    matched2 = retriever._find_matching_nodes(G, ["machine learning"])
    assert "Machine Learning" in matched2


def test_get_subgraph_chunks():
    """Kiểm tra duyệt graph thu thập chunk indices."""
    G = _build_sample_graph()
    retriever = GraphRetriever()

    # Từ Python (chunks [0,1]) → AI (chunks [1,2]) → ML (chunks [2])
    chunks = retriever._get_subgraph_chunks(G, {"Python"}, max_hops=2)

    assert 0 in chunks
    assert 1 in chunks
    assert 2 in chunks  # Qua 2 hops: Python → AI → ML


def test_get_related_facts():
    """Kiểm tra trích facts từ edges liên quan."""
    G = _build_sample_graph()
    retriever = GraphRetriever()

    facts = retriever._get_related_facts(G, {"AI"})

    assert any("sử dụng" in f for f in facts)
    assert any("lập trình cho" in f for f in facts)


def test_search_full_pipeline(mock_graph_deps, tmp_path, mocker):
    """Kiểm tra search end-to-end."""
    mock_llm, _ = mock_graph_deps
    mock_llm.chat_complete.return_value = '["Python"]'
    mocker.patch.object(GraphRetriever, "GRAPH_DIR", str(tmp_path))

    G = _build_sample_graph()
    with open(tmp_path / "search_col.gpickle", "wb") as f:
        pickle.dump(G, f)

    retriever = GraphRetriever()
    result = retriever.search("Python là gì?", "search_col")

    assert len(result["entities"]) > 0
    assert "Python" in result["matched_nodes"]
    assert len(result["facts"]) > 0
    assert len(result["chunk_indices"]) > 0
    assert "THỐNG KÊ" in result["graph_summary"]


def test_search_no_graph_returns_empty(mock_graph_deps, tmp_path, mocker):
    """Kiểm tra search trả về empty khi không có graph."""
    mocker.patch.object(GraphRetriever, "GRAPH_DIR", str(tmp_path))

    retriever = GraphRetriever()
    result = retriever.search("query", "missing_col")

    assert result["facts"] == []
    assert result["entities"] == []


def test_get_graph_summary(mock_graph_deps, tmp_path, mocker):
    """Kiểm tra tóm tắt graph."""
    mocker.patch.object(GraphRetriever, "GRAPH_DIR", str(tmp_path))

    G = _build_sample_graph()
    with open(tmp_path / "sum_col.gpickle", "wb") as f:
        pickle.dump(G, f)

    retriever = GraphRetriever()
    summary = retriever.get_graph_summary("sum_col")

    assert "Knowledge Graph" in summary
    assert "3 entities" in summary


def test_get_graph_summary_no_graph(mock_graph_deps, tmp_path, mocker):
    """Kiểm tra trả về chuỗi rỗng khi chưa có graph."""
    mocker.patch.object(GraphRetriever, "GRAPH_DIR", str(tmp_path))

    retriever = GraphRetriever()
    summary = retriever.get_graph_summary("missing")

    assert summary == ""
