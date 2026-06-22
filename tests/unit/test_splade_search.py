"""Tests for SpladeRetriever — neural sparse retrieval, cache, fallback."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_db(mocker):
    mock = MagicMock()
    mocker.patch("src.engine.retrieval.splade_search.db_instance", mock)
    return mock


from src.engine.retrieval.splade_search import SpladeRetriever  # noqa: E402


def test_search_returns_empty_when_models_fail(mock_db):
    """Model load failure → trả về [] thay vì crash."""
    retriever = SpladeRetriever(top_k=5)
    with patch.object(retriever, "_load_models", return_value=False):
        result = retriever.search("test query", "my-collection")
    assert result == []


def test_search_returns_empty_when_index_empty(mock_db):
    """Index rỗng → trả về []."""
    retriever = SpladeRetriever(top_k=5)
    # Pre-populate empty index (simulates collection with no docs)
    retriever._doc_vectors_cache["empty-col"] = []
    retriever._doc_mappings["empty-col"] = []

    with patch.object(retriever, "_load_models", return_value=True):
        result = retriever.search("test query", "empty-col")
    assert result == []


def test_index_populated_after_build(mock_db):
    """Cache miss → _build_or_get_vectors được gọi để build index."""
    retriever = SpladeRetriever(top_k=3)
    assert "new-col" not in retriever._doc_vectors_cache

    with patch.object(retriever, "_load_models", return_value=True):
        with patch.object(retriever, "_build_or_get_vectors", return_value=False) as mock_build:
            retriever.search("query", "new-col")
            mock_build.assert_called_once_with("new-col")


def test_clear_cache_removes_single_collection(mock_db):
    """clear_cache xóa đúng collection khỏi in-memory cache."""
    retriever = SpladeRetriever()
    retriever._doc_vectors_cache["col-a"] = [{}]
    retriever._doc_mappings["col-a"] = [{}]
    retriever._doc_vectors_cache["col-b"] = [{}]
    retriever._doc_mappings["col-b"] = [{}]

    retriever.clear_cache("col-a")

    assert "col-a" not in retriever._doc_vectors_cache
    assert "col-a" not in retriever._doc_mappings
    assert "col-b" in retriever._doc_vectors_cache


def test_clear_cache_multiple_collections(mock_db):
    """clear_cache xóa lần lượt nhiều collections."""
    retriever = SpladeRetriever()
    retriever._doc_vectors_cache["col-a"] = [{}]
    retriever._doc_mappings["col-a"] = [{}]
    retriever._doc_vectors_cache["col-b"] = [{}]
    retriever._doc_mappings["col-b"] = [{}]

    retriever.clear_cache("col-a")
    retriever.clear_cache("col-b")

    assert len(retriever._doc_vectors_cache) == 0
    assert len(retriever._doc_mappings) == 0


def test_encode_batch_returns_sparse_dicts(mock_db):
    """_encode_batch trả về list[dict] với non-negative weights."""
    import torch

    retriever = SpladeRetriever()
    retriever._models_loaded = True
    retriever._device = "cpu"

    # Mock tokenizer to return a MagicMock with .to() that returns tensor dict
    inputs_tensor = {
        "input_ids": torch.tensor([[101, 2054, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = inputs_tensor
    mock_inputs.__getitem__ = lambda self, key: inputs_tensor[key]
    mock_tokenizer = MagicMock(return_value=mock_inputs)

    logits = torch.zeros(1, 3, 30522)
    logits[0, :, 2054] = 2.0  # token 2054 gets positive weight
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(logits=logits)

    result = retriever._encode_batch(["test text"], mock_tokenizer, mock_model)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    # All weights must be non-negative (ReLU applied)
    assert all(v >= 0 for v in result[0].values())
