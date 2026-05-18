"""Unit tests for IngestionPipeline and ZenML steps."""

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
    step_extract_video, 
    step_semantic_chunking, 
    step_graph_extraction, 
    step_save_to_qdrantdb,
    IngestionPipeline
)


def test_step_extract_video(mock_loader):
    """Kiểm tra step 1: extract video."""
    result = step_extract_video("https://youtube.com/v123")
    assert result["metadata"]["video_id"] == "v123"
    mock_loader.load_video_data.assert_called_once()


def test_step_semantic_chunking(mocker):
    """Kiểm tra step 2: semantic chunking."""
    mock_chunker_cls = mocker.patch("src.engine.ingestion.pipeline.SemanticChunker")
    mock_chunker = MagicMock()
    mock_chunker.chunk_document.return_value = [{"content": "C1", "metadata": {}}]
    mock_chunker_cls.return_value = mock_chunker
    
    raw_data = {"metadata": {}, "transcript": []}
    result = step_semantic_chunking(raw_data, use_contextual=False)
    
    assert len(result) == 1
    assert result[0]["content"] == "C1"


def test_step_graph_extraction(mocker):
    """Kiểm tra step 3: graph extraction."""
    mock_ext_cls = mocker.patch("src.engine.ingestion.pipeline.GraphExtractor")
    mock_ext = MagicMock()
    mock_ext.process_chunk.return_value = {"content": "C1_processed", "metadata": {}}
    mock_ext_cls.return_value = mock_ext
    
    chunks = [{"content": "C1", "metadata": {}}]
    result = step_graph_extraction(chunks)
    
    assert result[0]["content"] == "C1_processed"


def test_step_save_to_qdrantdb(mock_db, mocker):
    """Kiểm tra step 4: save to qdrant."""
    # Mock KnowledgeGraphBuilder
    mocker.patch("src.engine.ingestion.pipeline.KnowledgeGraphBuilder")
    
    # Setup mock_db responses
    mock_db.get_or_create_collection.return_value = "test-collection"
    mock_db.embedding_model.encode.return_value = [[0.1, 0.2]]
    
    raw_data = {"metadata": {"video_id": "v123", "title": "Test Title"}}
    final_chunks = [{"content": "C1", "metadata": {"start_time": 0.0}}]
    
    result = step_save_to_qdrantdb(raw_data, final_chunks)
    
    assert result["status"] == "success"
    assert result["collection"] == "test-collection"
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
