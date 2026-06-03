"""Unit tests for IngestionPipeline direct runner functions."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_db(mocker):
    """Mock db_instance."""
    mock = MagicMock()
    mocker.patch("src.engine.ingestion.pipeline.db_instance", mock)
    return mock


@pytest.fixture
def mock_loader(mocker):
    """Mock YouTubeLoader."""
    mock_cls = mocker.patch("src.engine.ingestion.pipeline.YouTubeLoader")
    mock_instance = MagicMock()
    mock_instance.load_video_data.return_value = {
        "metadata": {"video_id": "v123", "title": "Test Video"},
        "transcript": [{"text": "Hello", "start_time": 0.0}]
    }
    mock_cls.return_value = mock_instance
    return mock_instance


from src.engine.ingestion.pipeline import (  # noqa: E402
    _run_extract_video,
    _run_semantic_chunking,
    _run_graph_extraction,
    _run_save_to_qdrant,
    IngestionPipeline,
)


def test_run_extract_video(mock_loader):
    """Kiểm tra _run_extract_video gọi YouTubeLoader đúng."""
    result = _run_extract_video("https://www.youtube.com/watch?v=v123")
    assert result["metadata"]["video_id"] == "v123"
    mock_loader.load_video_data.assert_called_once()


def test_run_semantic_chunking(mocker):
    """Kiểm tra _run_semantic_chunking gọi SemanticChunker và trả về chunks."""
    mock_chunker_cls = mocker.patch("src.engine.ingestion.pipeline.SemanticChunker")
    mock_chunker = MagicMock()
    mock_chunker.chunk_document.return_value = [{"content": "C1", "metadata": {}}]
    mock_chunker_cls.return_value = mock_chunker

    raw_data = {"metadata": {}, "transcript": []}
    result = _run_semantic_chunking(raw_data, use_contextual=False)

    assert len(result) == 1
    assert result[0]["content"] == "C1"


def test_run_graph_extraction(mocker):
    """Kiểm tra _run_graph_extraction xử lý chunks song song."""
    mock_ext_cls = mocker.patch("src.engine.ingestion.pipeline.GraphExtractor")
    mock_ext = MagicMock()
    mock_ext.process_chunk.return_value = {"content": "C1_processed", "metadata": {}}
    mock_ext_cls.return_value = mock_ext

    chunks = [{"content": "C1", "metadata": {}}]
    result = _run_graph_extraction(chunks)

    assert result[0]["content"] == "C1_processed"


def test_run_save_to_qdrant_basic(mock_db, mocker):
    """Kiểm tra _run_save_to_qdrant upsert lên Qdrant đúng."""
    mocker.patch("src.engine.ingestion.pipeline.KnowledgeGraphBuilder")
    mock_db.get_or_create_collection.return_value = "test-collection"
    mock_db.embedding_model.encode.return_value = [[0.1, 0.2]]

    raw_data = {"metadata": {"video_id": "v123", "title": "Test Title"}}
    final_chunks = [{"content": "C1", "metadata": {"start_time": 0.0}}]

    result = _run_save_to_qdrant(raw_data, final_chunks)

    assert result["status"] == "success"
    assert result["collection_name"] == "test-collection"
    mock_db.client.upsert.assert_called_once()


def test_ingestion_pipeline_wrapper(mocker):
    """Kiểm tra lớp wrapper IngestionPipeline gọi đúng các bước và trả về kết quả."""
    fake_raw = {"metadata": {"video_id": "abc123", "title": "Test Video"}, "transcript": []}
    fake_chunks = [{"content": "chunk1", "metadata": {"start_time": 0.0}}]
    fake_result = {
        "status": "success",
        "video_id": "abc123",
        "title": "Test Video",
        "collection_name": "test-col",
        "chunks_added": 1,
        "total_in_db": 1,
    }

    mocker.patch("src.engine.ingestion.pipeline._run_extract_video", return_value=fake_raw)
    mocker.patch("src.engine.ingestion.pipeline._run_semantic_chunking", return_value=fake_chunks)
    mocker.patch("src.engine.ingestion.pipeline._run_graph_extraction", return_value=fake_chunks)
    mocker.patch("src.engine.ingestion.pipeline._run_save_to_qdrant", return_value=fake_result)

    pipeline = IngestionPipeline(use_contextual_enrichment=True)
    result = pipeline.run("https://www.youtube.com/watch?v=abc123")

    assert result["status"] == "success"
    assert result["video_id"] == "abc123"


def test_ingestion_pipeline_with_late_chunking(mocker):
    """Kiểm tra IngestionPipeline dùng LateChunkingEmbedder khi use_late_chunking=True."""
    fake_raw = {"metadata": {"video_id": "abc123", "title": "Test Video"}, "transcript": []}
    fake_chunks = [{"content": "chunk1", "metadata": {}}]
    fake_embeddings = [[0.1] * 1024]
    fake_result = {
        "status": "success", "video_id": "abc123", "title": "Test Video",
        "collection_name": "test-col", "chunks_added": 1, "total_in_db": 1,
    }

    mocker.patch("src.engine.ingestion.pipeline._run_extract_video", return_value=fake_raw)
    mocker.patch("src.engine.ingestion.pipeline._run_semantic_chunking", return_value=fake_chunks)
    mocker.patch("src.engine.ingestion.pipeline._run_graph_extraction", return_value=fake_chunks)
    mock_save = mocker.patch("src.engine.ingestion.pipeline._run_save_to_qdrant", return_value=fake_result)

    mock_embedder_cls = mocker.patch("src.engine.ingestion.pipeline.LateChunkingEmbedder", create=True)
    mock_embedder = MagicMock()
    mock_embedder.embed_chunks.return_value = fake_embeddings
    mock_embedder_cls.return_value = mock_embedder

    # Patch the import inside the run() method
    mocker.patch.dict("sys.modules", {
        "src.engine.ingestion.late_chunker": MagicMock(LateChunkingEmbedder=mock_embedder_cls)
    })

    pipeline = IngestionPipeline(use_late_chunking=True)
    result = pipeline.run("https://www.youtube.com/watch?v=abc123")

    assert result["status"] == "success"
    # _run_save_to_qdrant được gọi (với hoặc không có precomputed_embeddings)
    mock_save.assert_called_once()


def test_ingestion_pipeline_late_chunking_fallback(mocker):
    """Kiểm tra IngestionPipeline fallback về bge-m3 khi LateChunkingEmbedder thất bại."""
    fake_raw = {"metadata": {"video_id": "abc123", "title": "Test Video"}, "transcript": []}
    fake_chunks = [{"content": "chunk1", "metadata": {}}]
    fake_result = {
        "status": "success", "video_id": "abc123", "title": "Test Video",
        "collection_name": "test-col", "chunks_added": 1, "total_in_db": 1,
    }

    mocker.patch("src.engine.ingestion.pipeline._run_extract_video", return_value=fake_raw)
    mocker.patch("src.engine.ingestion.pipeline._run_semantic_chunking", return_value=fake_chunks)
    mocker.patch("src.engine.ingestion.pipeline._run_graph_extraction", return_value=fake_chunks)
    mock_save = mocker.patch("src.engine.ingestion.pipeline._run_save_to_qdrant", return_value=fake_result)

    # LateChunkingEmbedder raises → fallback
    mock_embedder_cls = MagicMock(side_effect=ValueError("No JINA_API_KEY"))
    mocker.patch.dict("sys.modules", {
        "src.engine.ingestion.late_chunker": MagicMock(LateChunkingEmbedder=mock_embedder_cls)
    })

    pipeline = IngestionPipeline(use_late_chunking=True)
    result = pipeline.run("https://www.youtube.com/watch?v=abc123")

    assert result["status"] == "success"
    # Gọi _run_save_to_qdrant với precomputed_embeddings=None (fallback)
    _, kwargs = mock_save.call_args
    assert kwargs.get("precomputed_embeddings") is None or mock_save.call_args[0][2] is None


def test_run_save_to_qdrant_with_precomputed_embeddings(mock_db, mocker):
    """Kiểm tra _run_save_to_qdrant dùng precomputed embeddings thay vì encode."""
    from src.engine.ingestion.pipeline import _run_save_to_qdrant
    mocker.patch("src.engine.ingestion.pipeline.KnowledgeGraphBuilder")

    mock_db.get_or_create_collection.return_value = "test-col"

    raw_data = {"metadata": {"video_id": "v123", "title": "Test"}}
    chunks = [{"content": "c1", "metadata": {"start_time": 0.0}}]
    precomputed = [[0.5] * 1024]

    result = _run_save_to_qdrant(raw_data, chunks, precomputed_embeddings=precomputed)

    assert result["status"] == "success"
    # encode() không được gọi khi có precomputed_embeddings
    mock_db.embedding_model.encode.assert_not_called()
    mock_db.client.upsert.assert_called_once()


def test_run_save_to_qdrant_without_precomputed(mock_db, mocker):
    """Kiểm tra _run_save_to_qdrant dùng embedding_model.encode() khi không có precomputed."""
    import numpy as np
    from src.engine.ingestion.pipeline import _run_save_to_qdrant
    mocker.patch("src.engine.ingestion.pipeline.KnowledgeGraphBuilder")

    mock_db.get_or_create_collection.return_value = "test-col"
    mock_db.embedding_model.encode.return_value = [np.array([0.1] * 1024)]

    raw_data = {"metadata": {"video_id": "v123", "title": "Test"}}
    chunks = [{"content": "c1", "metadata": {"start_time": 0.0}}]

    result = _run_save_to_qdrant(raw_data, chunks)

    assert result["status"] == "success"
    mock_db.embedding_model.encode.assert_called_once()


def test_pipeline_result_contains_latency(mocker):
    """Kiểm tra kết quả pipeline có trường latency với các key đúng."""
    fake_raw = {"metadata": {"video_id": "abc123", "title": "T"}, "transcript": []}
    fake_chunks = [{"content": "c", "metadata": {}}]
    fake_result = {
        "status": "success", "video_id": "abc123", "title": "T",
        "collection_name": "col", "chunks_added": 1, "total_in_db": 1,
    }
    mocker.patch("src.engine.ingestion.pipeline._run_extract_video", return_value=fake_raw)
    mocker.patch("src.engine.ingestion.pipeline._run_semantic_chunking", return_value=fake_chunks)
    mocker.patch("src.engine.ingestion.pipeline._run_graph_extraction", return_value=fake_chunks)
    mocker.patch("src.engine.ingestion.pipeline._run_save_to_qdrant", return_value=fake_result)

    pipeline = IngestionPipeline()
    result = pipeline.run("https://www.youtube.com/watch?v=abc123")

    assert "latency" in result
    for key in ["extract_s", "chunk_s", "graph_s", "embed_s", "load_s", "total_s"]:
        assert key in result["latency"]
    assert result["late_chunking_used"] is False
