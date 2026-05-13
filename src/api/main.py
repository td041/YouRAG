from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import sys
import uuid

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.database import db_instance
from src.engine.ingestion.pipeline import IngestionPipeline
from src.engine.retrieval.hybrid_search import HybridRetriever
from src.engine.retrieval.graph_rag import GraphRetriever, KnowledgeGraphBuilder
from src.engine.ranking.cross_encoder import CrossEncoderReranker
from src.engine.generation.answer_generator import AnswerGenerator
from src.engine.generation.summarizer import VideoSummarizer
from src.engine.chat.history import ChatHistoryManager
from src.core.postgres import init_db
from src.core.logger import setup_logger
from src.cache.semantic_cache import SemanticCache

logger = setup_logger("YouRAG_API")
app = FastAPI(title="YouRAG Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# KHỞI TẠO SINGLETON MODELS (Load sẵn vào RAM/GPU)
# ─────────────────────────────────────────────
class AIStore:
    pipeline = None
    reranker = None
    generator = None
    summarizer = None
    graph_retriever = None
    cache = None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Đang khởi động Backend YouRAG...")
    init_db()
    AIStore.pipeline = IngestionPipeline()
    AIStore.reranker = CrossEncoderReranker()
    AIStore.generator = AnswerGenerator()
    AIStore.summarizer = VideoSummarizer()
    AIStore.graph_retriever = GraphRetriever()
    AIStore.cache = SemanticCache()
    logger.info("✅ TẤT CẢ MÔ HÌNH VÀ CACHING ĐÃ SẴN SÀNG!")

# ─────────────────────────────────────────────
# MODELS (Pydantic)
# ─────────────────────────────────────────────
class IngestRequest(BaseModel):
    url: str
    use_contextual: bool = False

class ChatRequest(BaseModel):
    query: str
    collection: str
    session_id: Optional[str] = None

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "online", "message": "YouRAG API is ready."}

@app.get("/collections")
def list_collections():
    collections_response = db_instance.client.get_collections()
    detailed_collections = []

    for c in collections_response.collections:
        try:
            records, _ = db_instance.client.scroll(
                collection_name=c.name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            if records and records[0].payload:
                meta = records[0].payload
                detailed_collections.append({
                    "name": c.name,
                    "title": meta.get("title", c.name),
                    "video_id": meta.get("video_id")
                })
            else:
                detailed_collections.append({"name": c.name, "title": c.name, "video_id": None})
        except Exception:
            detailed_collections.append({"name": c.name, "title": c.name, "video_id": None})

    return detailed_collections

@app.post("/ingest")
async def ingest_video(req: IngestRequest):
    try:
        AIStore.pipeline.use_contextual_enrichment = req.use_contextual
        result = AIStore.pipeline.run(req.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph/build/{collection}")
async def build_graph(collection: str):
    """Build (hoặc rebuild) Knowledge Graph cho một collection đã tồn tại."""
    try:
        builder = KnowledgeGraphBuilder()
        G = builder.build_graph(collection)
        # Invalidate cache trong GraphRetriever để load lại graph mới
        AIStore.graph_retriever._graph_cache.pop(collection, None)
        return {
            "status": "success",
            "collection": collection,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_rag(req: ChatRequest):
    try:
        session_id = req.session_id or str(uuid.uuid4())
        history_mgr = ChatHistoryManager(session_id=session_id, collection_name=req.collection)
        history_mgr.add_message(role="user", content=req.query)
        chat_history_str = history_mgr.format_for_prompt()

        # --- CHECK SEMANTIC CACHE ---
        cached_data = AIStore.cache.check_cache(req.query)
        if cached_data:
            answer = cached_data["answer"]
            sources = cached_data["sources"]
            facts = cached_data["facts"]
            
            history_mgr.add_message(role="assistant", content=answer)
            return {
                "answer": answer,
                "sources": sources,
                "facts": facts,
                "graph_entities": [], # entities not cached currently
                "session_id": session_id,
                "cached": True
            }

        # 1. Hybrid Search
        hybrid = HybridRetriever(top_k=10)
        candidates = hybrid.search(req.query, collection_name=req.collection)

        if not candidates:
            return {"answer": "Không tìm thấy thông tin phù hợp trong video này.", "sources": [], "facts": [], "session_id": session_id}

        # 2. Graph RAG Search (lấy graph_data thật)
        graph_data = AIStore.graph_retriever.search(req.query, collection_name=req.collection)

        # 3. Rerank
        final_chunks = AIStore.reranker.rerank(query=req.query, chunks=candidates, top_k=4)

        # 4. Global Summary
        global_summary = AIStore.summarizer.summarize(req.collection)

        # 5. Generate Answer
        answer = AIStore.generator.generate(
            query=req.query,
            retrieved_chunks=final_chunks,
            global_summary=global_summary,
            graph_facts=graph_data.get("facts", []),
            graph_summary=graph_data.get("graph_summary", ""),
            chat_history=chat_history_str
        )

        history_mgr.add_message(role="assistant", content=answer)

        from src.core.utils import format_timestamp
        sources = [
            f"{format_timestamp(c['metadata']['start_time'])}–{format_timestamp(c['metadata']['end_time'])}"
            for c in final_chunks
        ]

        # --- SAVE TO CACHE ---
        AIStore.cache.save_to_cache(
            query=req.query, 
            answer=answer, 
            sources=sources, 
            facts=graph_data.get("facts", [])
        )

        return {
            "answer": answer,
            "sources": sources,
            "facts": graph_data.get("facts", []),
            "graph_entities": graph_data.get("entities", []),
            "session_id": session_id,
            "cached": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
async def get_chat_history(session_id: str, collection: str):
    """Lấy lịch sử chat của một phiên để hiển thị lên UI khi load lại trang"""
    try:
        history_mgr = ChatHistoryManager(session_id=session_id, collection_name=collection)
        return history_mgr.get_history(limit=20)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_rag_stream(req: ChatRequest):
    """Endpoint trả về Streaming Response với Global Context."""
    try:
        session_id = req.session_id or str(uuid.uuid4())
        history_mgr = ChatHistoryManager(session_id=session_id, collection_name=req.collection)
        history_mgr.add_message(role="user", content=req.query)
        chat_history_str = history_mgr.format_for_prompt()

        # --- CHECK SEMANTIC CACHE ---
        cached_data = AIStore.cache.check_cache(req.query)
        if cached_data:
            def cached_generator():
                answer = cached_data["answer"]
                yield answer
                history_mgr.add_message(role="assistant", content=answer)
                
                yield f"\n\n__SOURCES__::{','.join(cached_data['sources'])}"
                if cached_data.get("facts"):
                    yield f"\n\n__FACTS__::{'|'.join(cached_data['facts'])}"
                yield f"\n\n__SESSION__::{session_id}"
                yield "\n\n__CACHED__::true"

            return StreamingResponse(cached_generator(), media_type="text/plain")

        # 1. Hybrid Search
        hybrid = HybridRetriever(top_k=10)
        candidates = hybrid.search(req.query, collection_name=req.collection)

        if not candidates:
            async def no_result():
                yield "Không tìm thấy thông tin phù hợp trong video này."
            return StreamingResponse(no_result(), media_type="text/plain")

        # 2. Graph RAG Search (lấy graph_data thật)
        graph_data = AIStore.graph_retriever.search(req.query, collection_name=req.collection)

        # 3. Rerank
        final_chunks = AIStore.reranker.rerank(query=req.query, chunks=candidates, top_k=4)

        # 4. Global Summary
        global_summary = AIStore.summarizer.summarize(req.collection)

        # 5. Streaming Generator
        def response_generator():
            full_response = ""
            for chunk in AIStore.generator.generate_stream(
                query=req.query,
                retrieved_chunks=final_chunks,
                global_summary=global_summary,
                graph_facts=graph_data.get("facts", []),
                graph_summary=graph_data.get("graph_summary", ""),
                chat_history=chat_history_str
            ):
                full_response += chunk
                yield chunk
            
            # Lưu lại tin nhắn AI sau khi stream xong
            history_mgr.add_message(role="assistant", content=full_response)

            from src.core.utils import format_timestamp
            sources = [
                f"{format_timestamp(c['metadata']['start_time'])}–{format_timestamp(c['metadata']['end_time'])}"
                for c in final_chunks
            ]
            yield f"\n\n__SOURCES__::{','.join(sources)}"

            if graph_data.get("facts"):
                yield f"\n\n__FACTS__::{'|'.join(graph_data['facts'])}"
            
            yield f"\n\n__SESSION__::{session_id}"
            
            # --- SAVE TO CACHE ---
            AIStore.cache.save_to_cache(
                query=req.query,
                answer=full_response,
                sources=sources,
                facts=graph_data.get("facts", [])
            )

        return StreamingResponse(response_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summarize/{collection}")
async def get_summary(collection: str):
    try:
        summary = AIStore.summarizer.summarize(collection)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summarize/stream/{collection}")
async def get_summary_stream(collection: str):
    """Endpoint tóm tắt video theo kiểu Streaming."""
    try:
        return StreamingResponse(
            AIStore.summarizer.summarize_stream(collection),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
