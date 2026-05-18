from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict as TypingDict
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
from src.api.auth import require_api_key

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

def _load_component(name: str, factory, attr: str) -> None:
    """Load một AI component vào AIStore; log lỗi thay vì crash toàn app."""
    try:
        setattr(AIStore, attr, factory())
        logger.info(f"  ✓ {name} ready")
    except Exception as e:
        logger.error(f"  ✗ {name} failed to load — feature will be unavailable: {e}")


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Đang khởi động Backend YouRAG...")

    try:
        init_db()
    except Exception as e:
        logger.error(f"[Startup] PostgreSQL init failed (non-critical): {e}")

    _load_component("IngestionPipeline", lambda: IngestionPipeline(), "pipeline")
    _load_component("CrossEncoderReranker", lambda: CrossEncoderReranker(), "reranker")
    _load_component("AnswerGenerator", lambda: AnswerGenerator(), "generator")
    _load_component("VideoSummarizer", lambda: VideoSummarizer(), "summarizer")
    _load_component("GraphRetriever", lambda: GraphRetriever(), "graph_retriever")
    _load_component("SemanticCache", lambda: SemanticCache(), "cache")

    loaded = [k for k, v in vars(AIStore).items() if not k.startswith("_") and v is not None]
    logger.info(f"✅ Startup complete. Loaded: {loaded}")

# ─────────────────────────────────────────────
# MODELS (Pydantic)
# ─────────────────────────────────────────────
class IngestRequest(BaseModel):
    url: str
    use_contextual: bool = False
    use_late_chunking: bool = False  # Jina Late Chunking (cần JINA_API_KEY)

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

# Collections dùng nội bộ — không hiển thị trong library
_INTERNAL_COLLECTIONS = {"semantic_cache"}

# Job store cho async ingest (in-memory, đủ cho single-node deployment)
_ingest_jobs: TypingDict[str, dict] = {}


def _run_ingest_job(job_id: str, url: str, use_contextual: bool, use_late_chunking: bool) -> None:
    """Background task: chạy ingest và cập nhật trạng thái job."""
    _ingest_jobs[job_id]["status"] = "running"
    try:
        pipeline = IngestionPipeline(
            use_contextual_enrichment=use_contextual,
            use_late_chunking=use_late_chunking,
        )
        result = pipeline.run(url)
        _ingest_jobs[job_id].update({"status": "done", "result": result})
    except Exception as e:
        logger.error(f"[IngestJob {job_id}] failed: {e}")
        _ingest_jobs[job_id].update({"status": "error", "error": str(e)})

@app.get("/collections")
def list_collections():
    collections_response = db_instance.client.get_collections()
    detailed_collections = []

    for c in collections_response.collections:
        if c.name in _INTERNAL_COLLECTIONS:
            continue
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

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str, _: str = Depends(require_api_key)):
    """Xóa một collection (video) khỏi Qdrant và graph store."""
    try:
        db_instance.client.delete_collection(collection_name)
        # Xóa knowledge graph nếu có
        import os
        graph_path = os.path.join("graph_store", f"{collection_name}.gpickle")
        if os.path.exists(graph_path):
            os.remove(graph_path)
        # Invalidate graph cache trong GraphRetriever
        if AIStore.graph_retriever:
            AIStore.graph_retriever._graph_cache.pop(collection_name, None)
        logger.info(f"🗑️ Deleted collection: {collection_name}")
        return {"status": "deleted", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_video(req: IngestRequest, background_tasks: BackgroundTasks, _: str = Depends(require_api_key)):
    """Khởi động ingest video bất đồng bộ. Trả về job_id để theo dõi tiến độ."""
    job_id = str(uuid.uuid4())
    _ingest_jobs[job_id] = {"status": "queued", "url": req.url}
    background_tasks.add_task(
        _run_ingest_job, job_id, req.url, req.use_contextual, req.use_late_chunking
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    """Kiểm tra trạng thái job ingest: queued → running → done / error."""
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    return job

@app.post("/graph/build/{collection}")
async def build_graph(collection: str, _: str = Depends(require_api_key)):
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
async def chat_rag(req: ChatRequest, _: str = Depends(require_api_key)):
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

        # 6. Generate Suggested Questions
        sq_prompt = f"Dựa trên câu hỏi '{req.query}' và câu trả lời '{answer}', hãy gợi ý đúng 3 câu hỏi ngắn gọn (mỗi câu dưới 12 từ) mà người dùng có thể hỏi tiếp theo. Trả về định dạng: Câu 1|Câu 2|Câu 3. Không gạch đầu dòng, không đánh số."
        try:
            sq_list = AIStore.generator.llm.chat_complete(sq_prompt, system="Bạn là trợ lý RAG.", max_tokens=100)
            suggestions = [s.strip() for s in sq_list.replace('\n', '').split("|") if s.strip()]
        except Exception:
            suggestions = []

        return {
            "answer": answer,
            "sources": sources,
            "facts": graph_data.get("facts", []),
            "graph_entities": graph_data.get("entities", []),
            "session_id": session_id,
            "cached": False,
            "suggestions": suggestions[:3]
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
async def chat_rag_stream(req: ChatRequest, _: str = Depends(require_api_key)):
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

            # Generate Suggested Questions
            sq_prompt = f"Dựa trên câu hỏi '{req.query}' và câu trả lời '{full_response}', hãy gợi ý đúng 3 câu hỏi ngắn gọn (mỗi câu dưới 12 từ) mà người dùng có thể hỏi tiếp theo. Trả về định dạng: Câu 1|Câu 2|Câu 3. Không gạch đầu dòng, không đánh số."
            try:
                sq_list = AIStore.generator.llm.chat_complete(sq_prompt, system="Bạn là trợ lý RAG.", max_tokens=100)
                # Clean up format issues from LLM just in case
                sq_clean = sq_list.replace('\n', '').strip()
                yield f"\n\n__SUGGESTIONS__::{sq_clean}"
            except Exception as e:
                logger.error(f"Error generating suggestions: {e}")

        return StreamingResponse(response_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summarize/{collection}")
async def get_summary(collection: str, _: str = Depends(require_api_key)):
    try:
        summary = AIStore.summarizer.summarize(collection)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summarize/stream/{collection}")
async def get_summary_stream(collection: str, _: str = Depends(require_api_key)):
    """Endpoint tóm tắt video theo kiểu Streaming."""
    try:
        return StreamingResponse(
            AIStore.summarizer.summarize_stream(collection),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
