import re
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from zenml import step, pipeline
    _ZENML_AVAILABLE = True
except ImportError:
    # ZenML không được cài trong production image — dùng direct runner
    _ZENML_AVAILABLE = False
    def step(fn=None, **_kwargs):
        """No-op decorator khi zenml không có."""
        return fn if fn is not None else lambda f: f
    def pipeline(fn=None, **_kwargs):
        return fn if fn is not None else lambda f: f

from src.engine.ingestion.youtube_loader import YouTubeLoader
from src.engine.ingestion.chunker import SemanticChunker
from src.engine.ingestion.graph_extractor import GraphExtractor
from src.engine.ingestion.contextual_enricher import ContextualEnricher
from src.core.database import db_instance
from src.core.logger import logger
from src.engine.retrieval.graph_rag import KnowledgeGraphBuilder

# -------------------------------------------------------------------
# 🧘 ZENML PIPELINE STEPS
# Mỗi @step là một "trạm kiểm soát" (checkpoint) độc lập. Nếu luồng bị lỗi ở trạm 3,
# lần sau chạy lại ZenML sẽ tự động lấy kết quả đã lưu ở trạm 2 để chạy tiếp,
# không bắt bạn phải cào lại Youtube từ đầu (tiết kiệm thời gian, chi phí API).
# -------------------------------------------------------------------

@step
def step_extract_video(youtube_url: str) -> Dict[str, Any]:
    """
    [STEP 1: EXTRACT YOUTUBE METADATA & TRANSCRIPT]
    - Nhiệm vụ: Tải transcript (nội dung chữ) và metadata (tiêu đề, tác giả) từ URL.
    - Tại sao tách riêng: Mạng tải Youtube rất hay chập chờn. Việc cô lập bước này 
      giúp nếu rớt mạng thì ZenML báo lỗi đúng cục này, dễ debug và không lặp lại tốn công.
    """
    logger.info(f"🎥 [ZenML Step 1] Loading video from: {youtube_url}")
    loader = YouTubeLoader()
    raw_data = loader.load_video_data(youtube_url)
    return raw_data

@step
def step_semantic_chunking(raw_data: Dict[str, Any], use_contextual: bool) -> List[Dict[str, Any]]:
    """
    [STEP 2: SEMANTIC CHUNKING & CONTEXTUAL ENRICHMENT]
    - Task 1 (Chunker): Dùng AI phân tích Vector để cắt chữ ra thành từng khúc (chunk). 
      Đảm bảo mỗi khúc trọn vẹn ngữ nghĩa (ví dụ: đang nói về Pin thì không bị cắt gãy sang Camera).
    - Task 2 (Enricher - Tùy chọn): AI đọc toàn bộ video, sau đó viết đoạn "ngữ cảnh" bổ sung 
      (Context Summary) chèn vào từng chunk. Đây là kỹ thuật SOTA của Anthropic, 
      giúp công cụ Retrieval sau này không bao giờ bị "mù" ngữ cảnh.
    """
    logger.info("🔪 [ZenML Step 2] Semantic Chunking & Contextual Enrichment...")
    llm_client = None
    if use_contextual:
        try:
            from src.engine.generation.llm_client import LLMClient
            llm_client = LLMClient()
        except Exception as e:
            logger.warning(f"Không thể khởi tạo LLMClient: {e}")

    chunker = SemanticChunker(
        percentile_threshold=15,
        pause_threshold_sec=1.5,
        min_chars_per_chunk=200,
        max_chars_per_chunk=2000,
        llm_client=llm_client
    )
    
    # 2.1 - Tiến hành cắt nhỏ văn bản
    chunks = chunker.chunk_document(raw_data["metadata"], raw_data["transcript"])
    
    # 2.2 - Tiến hành bơm ngữ cảnh (Nếu bật)
    if use_contextual and llm_client:
        enricher = ContextualEnricher(max_workers=5, llm_client=llm_client)
        full_text = " ".join(r["text"] for r in raw_data["transcript"])
        chunks = enricher.enrich(full_text, chunks)
        
    return chunks

@step
def step_graph_extraction(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [STEP 3: GRAPH & KEYWORD EXTRACTION]
    - Nhiệm vụ: Quét qua các chunk ở Bước 2 để rút trích các cụm danh từ, 
      nhóm thực thể đặc thù (Entities) (Ví dụ: iPhone 15, Apple, Chip A17).
    - Mục đích: Từ khóa này sẽ tiếp sức cho bộ máy BM25 (Sparse Retrieval), 
      giúp tìm kiếm từ khóa chính xác (Exact-match) hoạt động vượt trội hơn.
    """
    logger.info("🕸️ [ZenML Step 3] Graph & Keyword Extraction...")
    graph_rag = GraphExtractor()
    results = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {
            executor.submit(graph_rag.process_chunk, c): i
            for i, c in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"Lỗi GraphExtractor: {e}")
                results[idx] = chunks[idx]
    return results

@step
def step_save_to_qdrantdb(raw_data: Dict[str, Any], final_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    [STEP 4: VECTOR DATABASE UPSERT]
    - Nhiệm vụ: Đóng gói toàn bộ các cấu trúc Chunk (đã gắn vector, keywords, 
      thời gian) và bắn thẳng vào ngân hàng dữ liệu Qdrant.
    - Từ thời điểm này, toàn bộ data đã nằm chuẩn trong kiến trúc RAG, 
      sẵn sàng được gọi lên bởi LLaMa 3 để chat với người dùng.
    """
    logger.info("💾 [ZenML Step 4] Upserting to Qdrant...")
    video_id = raw_data["metadata"]["video_id"]
    title = raw_data["metadata"].get("title", "unknown")

    # Chuẩn hóa tên Collection để né lỗi quy định của Qdrant
    col_name = re.sub(r'[^a-z0-9]', '-', title.lower())
    col_name = re.sub(r'-+', '-', col_name).strip('-')
    if len(col_name) < 3:
        col_name = col_name.ljust(3, 'a')
    col_name = col_name[:63].strip('-')
    
    col_name = db_instance.get_or_create_collection(col_name)
    
    docs = [c["content"] for c in final_chunks]
    metas = [c["metadata"] for c in final_chunks]
    
    # Nhúng Vector bằng SentenceTransformer
    embeddings = db_instance.embedding_model.encode(docs)
    
    import uuid
    from qdrant_client.http import models

    points = []
    for i, (doc, meta, emb) in enumerate(zip(docs, metas, embeddings)):
        payload = {"text": doc}
        payload.update(meta)
        
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{i}")),
                vector=emb,
                payload=payload
            )
        )
        
    db_instance.client.upsert(
        collection_name=col_name,
        points=points
    )

    # Build Knowledge Graph ngay sau khi upsert xong
    try:
        builder = KnowledgeGraphBuilder()
        builder.build_graph(col_name)
        logger.info(f"🕸️ Knowledge Graph built for [{col_name}]")
    except Exception as e:
        logger.warning(f"⚠️ Knowledge Graph build failed (non-critical): {e}")

    return {
        "status": "success",
        "video_id": video_id,
        "title": title,
        "collection": col_name,
        "chunks_added": len(docs),
        "total_in_db": len(docs), # Qdrant return specific size takes different logic
    }

# -------------------------------------------------------------------
# 🎬 THE MASTER ORCHESTRATOR (ZENML PIPELINE)
# -------------------------------------------------------------------

@pipeline
def zenml_ingestion_pipeline(youtube_url: str, use_contextual: bool):
    """
    [MAIN PIPELINE]
    Đây là Trưởng luồng kết nối tất cả các bước (steps) lại với nhau thành mô hình DAG.
    Sản phẩm của bước này sẽ tự động chảy sang làm Đầu vào (Input) của bước kế tiếp.
    Trên bảng UI ZenML, nó sẽ tự động lập sơ đồ khối cực kỳ xịn sò.
    """
    raw_data = step_extract_video(youtube_url)
    chunks = step_semantic_chunking(raw_data, use_contextual)
    final_chunks = step_graph_extraction(chunks)
    result = step_save_to_qdrantdb(raw_data, final_chunks)
    return result


# -------------------------------------------------------------------
# 🏃 DIRECT RUNNER HELPERS (bypass ZenML for local/API use)
# -------------------------------------------------------------------

def _run_extract_video(youtube_url: str) -> Dict[str, Any]:
    logger.info(f"🎥 [Step 1] Loading video: {youtube_url}")
    loader = YouTubeLoader()
    return loader.load_video_data(youtube_url)


def _run_semantic_chunking(raw_data: Dict[str, Any], use_contextual: bool) -> List[Dict[str, Any]]:
    logger.info("🔪 [Step 2] Semantic Chunking...")
    llm_client = None
    if use_contextual:
        try:
            from src.engine.generation.llm_client import LLMClient
            llm_client = LLMClient()
        except Exception as e:
            logger.warning(f"LLMClient init failed: {e}")

    chunker = SemanticChunker(
        percentile_threshold=15,
        pause_threshold_sec=1.5,
        min_chars_per_chunk=200,
        max_chars_per_chunk=2000,
        llm_client=llm_client,
    )
    chunks = chunker.chunk_document(raw_data["metadata"], raw_data["transcript"])

    if use_contextual and llm_client:
        enricher = ContextualEnricher(max_workers=5, llm_client=llm_client)
        full_text = " ".join(r["text"] for r in raw_data["transcript"])
        chunks = enricher.enrich(full_text, chunks)

    return chunks


def _run_graph_extraction(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("🕸️ [Step 3] Graph Extraction...")
    extractor = GraphExtractor()
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {executor.submit(extractor.process_chunk, c): i for i, c in enumerate(chunks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"GraphExtractor error: {e}")
                results[idx] = chunks[idx]
    return results


def _run_save_to_qdrant(
    raw_data: Dict[str, Any],
    final_chunks: List[Dict[str, Any]],
    precomputed_embeddings: Optional[List] = None,
) -> Dict[str, Any]:
    import uuid
    from qdrant_client.http import models

    logger.info("💾 [Step 4] Upserting to Qdrant...")
    video_id = raw_data["metadata"]["video_id"]
    title = raw_data["metadata"].get("title", "unknown")

    col_name = re.sub(r'[^a-z0-9]', '-', title.lower())
    col_name = re.sub(r'-+', '-', col_name).strip('-')
    if len(col_name) < 3:
        col_name = col_name.ljust(3, 'a')
    col_name = col_name[:63].strip('-')
    col_name = db_instance.get_or_create_collection(col_name)

    docs = [c["content"] for c in final_chunks]
    metas = [c["metadata"] for c in final_chunks]

    if precomputed_embeddings is not None:
        logger.info("[Step 4] Dùng Late Chunking embeddings (Jina context-aware)")
        embeddings = precomputed_embeddings
    else:
        embeddings = db_instance.embedding_model.encode(docs, show_progress_bar=True)

    points = []
    for i, (doc, meta, emb) in enumerate(zip(docs, metas, embeddings)):
        payload = {"text": doc}
        payload.update(meta)
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}_{i}")),
            vector=emb.tolist() if hasattr(emb, "tolist") else emb,
            payload=payload,
        ))

    db_instance.client.upsert(collection_name=col_name, points=points)

    try:
        builder = KnowledgeGraphBuilder()
        builder.build_graph(col_name)
        logger.info(f"🕸️ Knowledge Graph built for [{col_name}]")
    except Exception as e:
        logger.warning(f"Knowledge Graph build failed (non-critical): {e}")

    return {
        "status": "success",
        "video_id": video_id,
        "title": title,
        "collection_name": col_name,
        "chunks_added": len(docs),
        "total_in_db": len(docs),
    }


# -------------------------------------------------------------------
# 🛡️ BACKWARD-COMPATIBLE WRAPPER FOR FASTAPI / STREAMLIT
# -------------------------------------------------------------------
class IngestionPipeline:
    def __init__(
        self,
        use_contextual_enrichment: bool = False,
        use_late_chunking: bool = False,
    ):
        self.use_contextual_enrichment = use_contextual_enrichment
        self.use_late_chunking = use_late_chunking

    def run(self, youtube_url: str, force_reingest: bool = False) -> Dict:
        import time
        logger.info(f"🚀 INGESTION STARTED: {youtube_url}")
        t0 = time.time()

        raw_data = _run_extract_video(youtube_url)
        t1 = time.time()

        chunks = _run_semantic_chunking(raw_data, self.use_contextual_enrichment)
        t2 = time.time()

        final_chunks = _run_graph_extraction(chunks)
        t3 = time.time()

        # Late Chunking: dùng Jina context-aware embeddings thay bge-m3 per-chunk
        precomputed_embeddings = None
        if self.use_late_chunking:
            try:
                from src.engine.ingestion.late_chunker import LateChunkingEmbedder
                embedder = LateChunkingEmbedder()
                texts = [c["content"] for c in final_chunks]
                precomputed_embeddings = embedder.embed_chunks(texts)
                logger.info("[LateChunking] ✅ Embeddings context-aware đã sẵn sàng")
            except Exception as e:
                logger.warning(f"[LateChunking] Thất bại, fallback về bge-m3: {e}")
                precomputed_embeddings = None
        t4 = time.time()

        result = _run_save_to_qdrant(raw_data, final_chunks, precomputed_embeddings)
        t5 = time.time()

        result["late_chunking_used"] = precomputed_embeddings is not None
        result["latency"] = {
            "extract_s": round(t1 - t0, 2),
            "chunk_s": round(t2 - t1, 2),
            "graph_s": round(t3 - t2, 2),
            "embed_s": round(t4 - t3, 2),
            "load_s": round(t5 - t4, 2),
            "total_s": round(t5 - t0, 2),
        }

        logger.info(
            f"✅ INGESTION COMPLETED in {result['latency']['total_s']}s — "
            f"{result['chunks_added']} chunks → [{result['collection_name']}]"
        )
        return result
