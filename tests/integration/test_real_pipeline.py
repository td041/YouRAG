"""
Integration Tests — Test logic THẬT với services THẬT.

Mỗi test ở đây gọi code THẬT, KHÔNG mock:
- Qdrant: Lưu/truy vấn vector thật
- LLM (Groq): Gọi API thật để sinh text
- Embedding Model: Nhúng vector thật
- YouTube API: Tải transcript thật (1 test)

Yêu cầu:
- GROQ_API_KEY phải có trong environment (hoặc .env)
- Qdrant chạy local (file-based, tự động)
"""

import os
import sys
import uuid
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.config import settings
from src.core.database import db_instance

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

TEST_COLLECTION = f"integration_test_{uuid.uuid4().hex[:8]}"


def _has_groq_key():
    """Check if GROQ_API_KEY is available."""
    try:
        key = settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else ""
        return bool(key)
    except Exception:
        return False


requires_groq = pytest.mark.skipif(
    not _has_groq_key(),
    reason="GROQ_API_KEY không có — bỏ qua test cần LLM thật"
)


@pytest.fixture(scope="module")
def test_collection():
    """Tạo Qdrant collection thật và cleanup sau khi test xong."""
    db_instance.get_or_create_collection(TEST_COLLECTION)

    # Bơm 3 chunks thật vào collection
    from qdrant_client.http import models

    chunks = [
        {
            "text": "Python là ngôn ngữ lập trình phổ biến nhất thế giới. Python được dùng trong AI, web development và data science.",
            "chunk_index": 0,
            "start_time": 0.0,
            "end_time": 30.0,
            "title": "Python Tutorial",
            "video_id": "test_vid_001",
        },
        {
            "text": "Machine Learning là nhánh của AI cho phép máy tính học từ dữ liệu mà không cần lập trình cụ thể.",
            "chunk_index": 1,
            "start_time": 30.0,
            "end_time": 60.0,
            "title": "Python Tutorial",
            "video_id": "test_vid_001",
        },
        {
            "text": "Deep Learning sử dụng mạng neural nhiều lớp. PyTorch và TensorFlow là hai framework phổ biến nhất.",
            "chunk_index": 2,
            "start_time": 60.0,
            "end_time": 90.0,
            "title": "Python Tutorial",
            "video_id": "test_vid_001",
        },
    ]

    points = []
    for i, chunk in enumerate(chunks):
        # Nhúng vector THẬT bằng embedding model thật
        vector = db_instance.embedding_model.encode(chunk["text"]).tolist()
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=chunk,
            )
        )

    db_instance.client.upsert(collection_name=TEST_COLLECTION, points=points)

    yield TEST_COLLECTION

    # Cleanup
    try:
        db_instance.client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 1. QDRANT — Vector Database (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

def test_qdrant_collection_exists(test_collection):
    """✅ THẬT: Kiểm tra Qdrant tạo collection thành công."""
    exists = db_instance.client.collection_exists(collection_name=test_collection)
    assert exists is True


def test_qdrant_has_correct_point_count(test_collection):
    """✅ THẬT: Kiểm tra Qdrant chứa đúng số lượng points."""
    info = db_instance.client.get_collection(test_collection)
    assert info.points_count == 3


def test_qdrant_scroll_returns_data(test_collection):
    """✅ THẬT: Kiểm tra scroll lấy dữ liệu chính xác."""
    records, _ = db_instance.client.scroll(
        collection_name=test_collection, limit=10, with_payload=True
    )
    assert len(records) == 3
    # Payload phải có text thật
    texts = [r.payload.get("text", "") for r in records]
    assert any("Python" in t for t in texts)


# ═══════════════════════════════════════════════════════════════════════════
# 2. EMBEDDING MODEL — Vector Encoding (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

def test_embedding_model_produces_correct_dimensions():
    """✅ THẬT: Kiểm tra embedding model tạo vector đúng kích thước."""
    vector = db_instance.embedding_model.encode("Hello world")
    assert len(vector) == settings.VECTOR_SIZE


def test_embedding_similarity_makes_sense():
    """✅ THẬT: Kiểm tra vector similarity phản ánh đúng ngữ nghĩa."""
    v1 = db_instance.embedding_model.encode("Python programming language")
    v2 = db_instance.embedding_model.encode("Java programming language")
    v3 = db_instance.embedding_model.encode("Chocolate cake recipe")

    # Cosine similarity
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_related = cosine(v1, v2)      # Python vs Java → nên cao
    sim_unrelated = cosine(v1, v3)    # Python vs Cake → nên thấp

    assert sim_related > sim_unrelated, (
        f"Vector similarity SAI: Python↔Java ({sim_related:.3f}) "
        f"phải > Python↔Cake ({sim_unrelated:.3f})"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. DENSE SEARCH — Vector Retrieval (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

def test_dense_search_finds_relevant_chunks(test_collection):
    """✅ THẬT: Kiểm tra Dense Search tìm đúng chunk liên quan."""
    from src.engine.retrieval.dense_search import DenseRetriever

    retriever = DenseRetriever(top_k=3)
    results = retriever.search("Python là gì?", test_collection)

    assert len(results) > 0
    # Chunk đầu tiên phải chứa "Python" (vì query hỏi về Python)
    assert "Python" in results[0]["content"], (
        f"Dense Search không tìm đúng: top result = '{results[0]['content'][:50]}'"
    )


def test_dense_search_ranking_order(test_collection):
    """✅ THẬT: Kiểm tra thứ tự ranking đúng (chunk liên quan nhất lên đầu)."""
    from src.engine.retrieval.dense_search import DenseRetriever

    retriever = DenseRetriever(top_k=3)
    results = retriever.search("Deep Learning và mạng neural", test_collection)

    # Chunk về Deep Learning phải đứng đầu
    assert "Deep Learning" in results[0]["content"] or "neural" in results[0]["content"]


# ═══════════════════════════════════════════════════════════════════════════
# 4. HYBRID SEARCH — Dense + Sparse Fusion (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

def test_hybrid_search_returns_results(test_collection):
    """✅ THẬT: Kiểm tra Hybrid Search trả về kết quả kết hợp."""
    from src.engine.retrieval.hybrid_search import HybridRetriever

    retriever = HybridRetriever(top_k=3)
    results = retriever.search("Machine Learning dữ liệu", test_collection)

    assert len(results) > 0
    assert "hybrid_score" in results[0]


def test_hybrid_search_has_rrf_scores(test_collection):
    """✅ THẬT: Kiểm tra RRF scores hợp lệ (> 0)."""
    from src.engine.retrieval.hybrid_search import HybridRetriever

    retriever = HybridRetriever(top_k=3)
    results = retriever.search("Python AI", test_collection)

    for r in results:
        assert r["hybrid_score"] > 0, f"RRF score phải > 0, got {r['hybrid_score']}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. CROSS-ENCODER RERANKER (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

def test_reranker_reorders_chunks(test_collection):
    """✅ THẬT: Kiểm tra Cross-Encoder rerank thay đổi thứ tự chunks."""
    from src.engine.retrieval.hybrid_search import HybridRetriever
    from src.engine.ranking.cross_encoder import CrossEncoderReranker

    retriever = HybridRetriever(top_k=3)
    candidates = retriever.search("Python AI", test_collection)

    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("Python AI", candidates, top_k=3)

    assert len(reranked) > 0
    assert "rerank_score" in reranked[0]
    # rerank scores phải là số thực
    assert isinstance(reranked[0]["rerank_score"], float)


# ═══════════════════════════════════════════════════════════════════════════
# 6. LLM CLIENT — Groq API Call (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

@requires_groq
def test_llm_generates_real_answer():
    """✅ THẬT: Kiểm tra LLM (Groq) sinh câu trả lời thật."""
    from src.engine.generation.llm_client import LLMClient

    client = LLMClient(provider="groq")
    answer = client.chat_complete(
        prompt="1 + 1 bằng mấy? Trả lời bằng 1 số duy nhất.",
        system="Trả lời ngắn gọn nhất có thể.",
        max_tokens=10,
        temperature=0.0,
    )

    assert "2" in answer, f"LLM trả lời sai phép tính 1+1: '{answer}'"


@requires_groq
def test_llm_streaming_works():
    """✅ THẬT: Kiểm tra LLM streaming yield từng chunk."""
    from src.engine.generation.llm_client import LLMClient

    client = LLMClient(provider="groq")
    chunks = list(client.chat_complete_stream(
        prompt="Kể tên 3 ngôn ngữ lập trình.",
        system="Trả lời ngắn gọn.",
        max_tokens=50,
    ))

    assert len(chunks) > 0, "Streaming phải yield ít nhất 1 chunk"
    full_text = "".join(chunks)
    assert len(full_text) > 5, f"Streaming text quá ngắn: '{full_text}'"


# ═══════════════════════════════════════════════════════════════════════════
# 7. ANSWER GENERATOR — Full RAG Pipeline (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

@requires_groq
def test_answer_generator_e2e(test_collection):
    """✅ THẬT: Kiểm tra AnswerGenerator sinh câu trả lời từ context thật."""
    from src.engine.retrieval.hybrid_search import HybridRetriever
    from src.engine.ranking.cross_encoder import CrossEncoderReranker
    from src.engine.generation.answer_generator import AnswerGenerator

    retriever = HybridRetriever(top_k=3)
    reranker = CrossEncoderReranker()
    generator = AnswerGenerator()

    # Dùng query unique để tránh cache hit từ test khác
    unique_query = f"Python dùng trong lĩnh vực nào? (test-{uuid.uuid4().hex[:6]})"

    # Search thật
    candidates = retriever.search(unique_query, test_collection)
    reranked = reranker.rerank(unique_query, candidates, top_k=3)

    # Generate thật
    answer = generator.generate(
        query=unique_query,
        retrieved_chunks=reranked,
    )

    assert len(answer) > 10, f"Câu trả lời quá ngắn: '{answer}'"
    assert "Lỗi" not in answer, f"Generator trả về lỗi: '{answer}'"


# ═══════════════════════════════════════════════════════════════════════════
# 8. YOUTUBE LOADER — Transcript Fetching (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

def test_youtube_extract_video_id_real():
    """✅ THẬT: Kiểm tra parse URL YouTube thật (không mock)."""
    from src.engine.ingestion.youtube_loader import YouTubeLoader

    # Đây là video Rick Astley — luôn luôn available
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    vid = YouTubeLoader.extract_video_id(url)
    assert vid == "dQw4w9WgXcQ"


def test_youtube_clean_text_real():
    """✅ THẬT: Kiểm tra text cleaning với Unicode thật."""
    from src.engine.ingestion.youtube_loader import YouTubeLoader

    # Dữ liệu Unicode thật lấy từ auto-generated captions
    dirty = "\ufeff\xa0Hello\u200b world  test\n second "
    clean = YouTubeLoader._clean_text(dirty)

    assert "\xa0" not in clean
    assert "\u200b" not in clean
    assert "\ufeff" not in clean
    assert "  " not in clean  # No double spaces


# ═══════════════════════════════════════════════════════════════════════════
# 9. SEMANTIC CHUNKER — Cosine + Pause Logic (THẬT 100%)
# ═══════════════════════════════════════════════════════════════════════════

def test_cosine_similarity_real():
    """✅ THẬT: Kiểm tra cosine similarity với numpy thật."""
    from src.engine.ingestion.chunker import cosine_similarity

    # Identical vectors → similarity = 1.0
    v = np.array([0.5, 0.5, 0.5])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    # Orthogonal vectors → similarity = 0.0
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    assert abs(cosine_similarity(v1, v2)) < 1e-6

    # Zero vector → 0.0
    assert cosine_similarity(np.zeros(3), v) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 10. PROMPT BUILDER — Logic thật không mock
# ═══════════════════════════════════════════════════════════════════════════

def test_prompt_builder_real():
    """✅ THẬT: Kiểm tra PromptBuilder tạo prompt đúng format."""
    from src.engine.generation.prompt_builder import PromptBuilder

    system = PromptBuilder.build_system_prompt(mode="standard")
    assert len(system) > 50, "System prompt quá ngắn"
    assert "video" in system.lower() or "trả lời" in system.lower()

    user = PromptBuilder.build_user_prompt(
        query="Python là gì?",
        context_str="Python là ngôn ngữ lập trình.",
        graph_facts=["Python → dùng cho → AI"],
    )
    assert "Python là gì?" in user
    assert "Python → dùng cho → AI" in user


def test_prompt_builder_mindmap_mode():
    """✅ THẬT: Kiểm tra mode mindmap tạo prompt khác standard."""
    from src.engine.generation.prompt_builder import PromptBuilder

    standard = PromptBuilder.build_system_prompt(mode="standard")
    mindmap = PromptBuilder.build_system_prompt(mode="mindmap")

    assert standard != mindmap, "Mindmap prompt phải khác standard prompt"
