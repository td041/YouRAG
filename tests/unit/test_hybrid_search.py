"""Tests for HybridRetriever — RRF fusion formula, ranking, edge cases."""

import pytest

# Mock DenseRetriever and SparseRetriever BEFORE importing HybridRetriever
# because __init__ constructs them immediately
@pytest.fixture(autouse=True)
def mock_retrievers(mocker):
    """Block DenseRetriever and SparseRetriever instantiation."""
    mocker.patch("src.engine.retrieval.hybrid_search.DenseRetriever")
    mocker.patch("src.engine.retrieval.hybrid_search.SparseRetriever")


from src.engine.retrieval.hybrid_search import HybridRetriever  # noqa: E402


def _make_doc(doc_id: str, content: str = "test content") -> dict:
    """Helper to create a mock retrieved document."""
    return {"id": doc_id, "content": content, "metadata": {}, "distance": 0.9}


def test_rrf_score_dense_only():
    """Kiểm tra RRF score khi chỉ có dense results (alpha=1.0)."""
    retriever = HybridRetriever(top_k=3, rrf_k=60, alpha=1.0)
    retriever.dense.search.return_value = [_make_doc("a")]
    retriever.sparse.search.return_value = []

    results = retriever.search("test query", "my-collection")

    assert len(results) == 1
    assert results[0]["id"] == "a"
    # alpha=1.0 means full weight to dense; doc at rank 1 → score = 1.0 * (1/(60+1))
    expected_score = round(1.0 * (1.0 / (60 + 1)), 5)
    assert results[0]["hybrid_score"] == expected_score


def test_rrf_score_sparse_only():
    """Kiểm tra RRF score khi chỉ có sparse results (alpha=0.0)."""
    retriever = HybridRetriever(top_k=3, rrf_k=60, alpha=0.0)
    retriever.dense.search.return_value = []
    retriever.sparse.search.return_value = [_make_doc("b")]

    results = retriever.search("test query", "my-collection")

    assert len(results) == 1
    assert results[0]["id"] == "b"
    expected_score = round(1.0 * (1.0 / (60 + 1)), 5)
    assert results[0]["hybrid_score"] == expected_score


def test_rrf_both_same_doc_combined_score():
    """Kiểm tra doc xuất hiện ở cả dense và sparse được cộng điểm từ cả hai."""
    retriever = HybridRetriever(top_k=3, rrf_k=60, alpha=0.5)
    retriever.dense.search.return_value = [_make_doc("x")]
    retriever.sparse.search.return_value = [_make_doc("x")]

    results = retriever.search("test query", "my-collection")

    assert len(results) == 1
    # Both at rank 1: 0.5*(1/61) + 0.5*(1/61) = 1/61
    expected_score = round(1.0 / 61, 5)
    assert results[0]["hybrid_score"] == expected_score


def test_rrf_ranking_order_non_overlap():
    """Kiểm tra thứ tự xếp hạng khi doc xuất hiện ở cả hai retrievers với rank khác nhau.

    doc "winner" xuất hiện ở rank 1 dense + rank 1 sparse → score cao nhất.
    doc "loser" chỉ xuất hiện ở rank 1 dense (không có sparse) → score thấp hơn.
    """
    retriever = HybridRetriever(top_k=5, rrf_k=60, alpha=0.5)
    # "winner" at rank 1 in dense AND rank 1 in sparse → double score
    # "loser" at rank 1 in dense only (not in sparse)
    retriever.dense.search.return_value = [_make_doc("loser"), _make_doc("winner")]
    retriever.sparse.search.return_value = [_make_doc("winner")]

    results = retriever.search("test query", "my-collection")

    ids = [r["id"] for r in results]
    # "winner" has both dense (rank 2) and sparse (rank 1) scores
    # dense rank 2: 0.5 * (1/62) = 0.00806
    # sparse rank 1: 0.5 * (1/61) = 0.00820
    # total = 0.01626
    # "loser" has only dense rank 1: 0.5 * (1/61) = 0.00820
    # "winner" total > "loser" total
    assert ids[0] == "winner"


def test_returns_empty_when_no_results():
    """Kiểm tra trả về [] khi cả dense và sparse đều rỗng."""
    retriever = HybridRetriever(top_k=5)
    retriever.dense.search.return_value = []
    retriever.sparse.search.return_value = []

    results = retriever.search("test query", "my-collection")
    assert results == []


def test_top_k_limits_output():
    """Kiểm tra top_k giới hạn số lượng kết quả trả về."""
    retriever = HybridRetriever(top_k=3, rrf_k=60, alpha=1.0)
    # 6 dense results, 0 sparse
    retriever.dense.search.return_value = [_make_doc(f"doc{i}") for i in range(6)]
    retriever.sparse.search.return_value = []

    results = retriever.search("test query", "my-collection")
    assert len(results) == 3


def test_hybrid_score_key_present_distance_removed():
    """Kiểm tra kết quả có 'hybrid_score' và không còn 'score' hay 'distance'."""
    retriever = HybridRetriever(top_k=5)
    retriever.dense.search.return_value = [_make_doc("doc1")]
    retriever.sparse.search.return_value = []

    results = retriever.search("test query", "my-collection")

    assert len(results) == 1
    assert "hybrid_score" in results[0]
    assert "distance" not in results[0]
    assert "score" not in results[0]
