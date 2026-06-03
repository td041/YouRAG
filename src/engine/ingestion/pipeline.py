import re
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.engine.ingestion.youtube_loader import YouTubeLoader
from src.engine.ingestion.chunker import SemanticChunker
from src.engine.ingestion.graph_extractor import GraphExtractor
from src.engine.ingestion.contextual_enricher import ContextualEnricher
from src.core.database import db_instance
from src.core.logger import logger
from src.engine.retrieval.graph_rag import KnowledgeGraphBuilder

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
