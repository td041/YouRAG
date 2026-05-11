import pytest
import os
import uuid
import sys
from qdrant_client.http import models

# Ensure absolute import path works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.database import db_instance
from src.core.config import settings
from src.engine.retrieval.graph_rag import KnowledgeGraphBuilder, GraphRetriever

TEST_COLLECTION = "test_graph_collection"

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """Tạo collection mẫu và xóa dọn dẹp sau khi test xong."""
    # Setup
    print("\n[SETUP] Khởi tạo Qdrant Collection Test...")
    db_instance.get_or_create_collection(TEST_COLLECTION)
    
    # Bơm 1 chunk chứa dữ liệu đồ thị rõ ràng
    # Chú ý: vector cần có size giống VECTOR_SIZE của cấu hình (mặc định BGE-M3 là 1024)
    dummy_vector = [0.0] * settings.VECTOR_SIZE
    
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=dummy_vector,
        payload={
            "text": "Tony Stark chế tạo Iron Man suit. Tony Stark là CEO của Stark Industries. Pepper Potts là thư ký của Tony Stark.",
            "chunk_index": 0,
            "start_time": 0.0
        }
    )
    
    db_instance.client.upsert(
        collection_name=TEST_COLLECTION,
        points=[point]
    )
    
    yield  # Test chạy ở đây
    
    # Teardown
    print("\n[TEARDOWN] Xóa dọn dẹp Qdrant và files...")
    try:
        db_instance.client.delete_collection(TEST_COLLECTION)
    except Exception as e:
        print(f"Lỗi khi xóa collection: {e}")
        
    builder = KnowledgeGraphBuilder()
    graph_path = os.path.join(builder.GRAPH_DIR, f"{TEST_COLLECTION}.gpickle")
    triples_path = os.path.join(builder.GRAPH_DIR, f"{TEST_COLLECTION}_triples.json")
    
    if os.path.exists(graph_path):
        os.remove(graph_path)
    if os.path.exists(triples_path):
        os.remove(triples_path)
        
    # Xóa bộ nhớ đệm
    retriever = GraphRetriever()
    retriever._graph_cache.pop(TEST_COLLECTION, None)

@pytest.mark.integration
def test_real_graph_rag_pipeline():
    """Chạy một luồng End-to-End thực tế với LLM thật."""
    
    # Đảm bảo bài test fail chứ không skip nếu thiếu Key
    assert settings.GROQ_API_KEY is not None, "BẮT BUỘC: Phải cung cấp GROQ_API_KEY trong cấu hình CI để test bằng LLM thật."

    print("\n[TEST] 1. Xây dựng Knowledge Graph (build_graph)...")
    builder = KnowledgeGraphBuilder()
    G = builder.build_graph(TEST_COLLECTION)
    
    assert G is not None, "Đồ thị phải được khởi tạo thành công."
    assert G.number_of_nodes() > 0, "Đồ thị không được rỗng, LLM phải trích xuất được ít nhất 1 node."
    
    # Kiểm tra xem file lưu xuống ổ cứng chưa
    graph_path = os.path.join(builder.GRAPH_DIR, f"{TEST_COLLECTION}.gpickle")
    assert os.path.exists(graph_path), f"File đồ thị chưa được lưu tại {graph_path}"

    print(f"\n[TEST] -> LLM đã tạo ra {G.number_of_nodes()} nodes và {G.number_of_edges()} edges.")
    
    # Kiểm tra các node thực tế có chứa Tony Stark không (case-insensitive checking cho an toàn)
    node_names = [n.lower() for n in G.nodes]
    assert any("tony" in n for n in node_names), "Đồ thị không tìm thấy entity 'Tony Stark' như dự kiến."

    print("\n[TEST] 2. Truy vấn Graph RAG (search)...")
    retriever = GraphRetriever()
    query = "Mối quan hệ giữa Tony Stark và Iron Man suit là gì?"
    
    result = retriever.search(query=query, collection_name=TEST_COLLECTION)
    
    assert isinstance(result, dict), "Search phải trả về một Dictionary"
    
    entities = result.get("entities", [])
    print(f"\n[TEST] -> LLM trích xuất Entities từ câu hỏi: {entities}")
    assert len(entities) > 0, "LLM không nhận diện được Entity nào từ câu hỏi."
    assert any("tony" in e.lower() or "iron" in e.lower() for e in entities), "Không tìm thấy Entity 'Tony Stark' hoặc 'Iron Man' từ câu hỏi."
    
    facts = result.get("facts", [])
    print(f"\n[TEST] -> Sự kiện (Facts) trích xuất được: {facts}")
    assert len(facts) > 0, "Không tìm thấy Sự kiện/Mối quan hệ nào liên kết với Entity."
    assert any("tony" in f.lower() for f in facts), "Các sự kiện không chứa thông tin về Tony Stark."
    
    chunk_indices = result.get("chunk_indices", set())
    assert 0 in chunk_indices, "Không tìm thấy chunk_index gốc (0) sau khi duyệt đồ thị."
    
    print("\n✅ TẤT CẢ Assertions đã Pass! Graph RAG hoạt động hoàn hảo với AI thật.")
