from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict as TypingDict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import sys
import uuid
import time
import threading

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
from src.core.config import settings
from src.core.postgres import init_db
from src.core.logger import setup_logger
from src.cache.semantic_cache import SemanticCache
from src.api.auth import require_api_key
from src.core.langfuse_client import get_langfuse
import re
import json

_YOUTUBE_URL_RE = re.compile(
    r"^(https?://)?(www\.)?"
    r"(youtube\.com/(watch\?.*v=|shorts/|embed/)|youtu\.be/)"
    r"[A-Za-z0-9_\-]{11}"
)

logger = setup_logger("YouRAG_API")

# Rate limiter — dùng IP address làm key
limiter = Limiter(key_func=get_remote_address)

# Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _prometheus_available = True
except ImportError:
    _prometheus_available = False

app = FastAPI(title="YouRAG Backend API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Expose /metrics for Prometheus scraping
if _prometheus_available:
    Instrumentator().instrument(app).expose(app)

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
    hybrid_retriever = None  # singleton — BM25 index cached across requests
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
    _load_component("HybridRetriever", lambda: HybridRetriever(top_k=10), "hybrid_retriever")
    _load_component("SemanticCache", lambda: SemanticCache(), "cache")

    loaded = [k for k, v in vars(AIStore).items() if not k.startswith("_") and v is not None]
    logger.info(f"✅ Startup complete. Loaded: {loaded}")

    # Start Redis Stream worker for ingest queue
    t = threading.Thread(target=_stream_worker_loop, daemon=True, name="stream-worker")
    t.start()

# ─────────────────────────────────────────────
# MODELS (Pydantic)
# ─────────────────────────────────────────────
class IngestRequest(BaseModel):
    url: str
    use_contextual: bool = False
    use_late_chunking: bool = True  # Jina Late Chunking (cần JINA_API_KEY, fallback bge-m3 nếu lỗi)

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


@app.get("/health")
def health_check():
    """Health check endpoint for Railway/Vercel/Docker. Returns 200 if all critical services are up."""
    checks: TypingDict[str, str] = {}
    healthy = True

    # Qdrant
    try:
        db_instance.client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
        healthy = False

    # Redis
    try:
        r = _get_redis()
        if r:
            r.ping()
            checks["redis"] = "ok"
        else:
            checks["redis"] = "unavailable (in-memory fallback active)"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # LLM models loaded
    checks["generator"] = "ok" if AIStore.generator else "not loaded"
    checks["reranker"] = "ok" if AIStore.reranker else "not loaded"

    status_code = 200 if healthy else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if healthy else "degraded", "checks": checks}
    )

# Collections dùng nội bộ — không hiển thị trong library
_INTERNAL_COLLECTIONS = {"semantic_cache"}

# ── Redis job store ──────────────────────────────────────────────────────────
# Fallback về in-memory dict nếu Redis không kết nối được (dev mode)
_JOB_TTL = 60 * 60 * 24  # 24 giờ
_ingest_jobs_fallback: TypingDict[str, dict] = {}
_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            import redis as redis_lib
            r = redis_lib.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
            r.ping()
            _redis_client = r
            logger.info("✅ Redis job store connected")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable ({e}) — using in-memory job store")
    return _redis_client

def _job_set(job_id: str, data: dict) -> None:
    r = _get_redis()
    if r:
        r.setex(f"ingest_job:{job_id}", _JOB_TTL, json.dumps(data))
    else:
        _ingest_jobs_fallback[job_id] = data

def _job_get(job_id: str) -> Optional[dict]:
    r = _get_redis()
    if r:
        raw = r.get(f"ingest_job:{job_id}")
        return json.loads(raw) if raw else None
    return _ingest_jobs_fallback.get(job_id)

def _job_update(job_id: str, patch: dict) -> None:
    data = _job_get(job_id) or {}
    data.update(patch)
    _job_set(job_id, data)


# ── Redis Streams worker ──────────────────────────────────────────────────────
_STREAM_KEY = "yourag:ingest"
_STREAM_GROUP = "workers"
_STREAM_CONSUMER = "worker-1"
_stream_worker_active = False


def _process_stream_msg(r, msg_id: str, fields: dict) -> None:
    """Execute one ingest job from the stream then acknowledge it."""
    job_id = fields.get("job_id", "")
    url = fields.get("url", "")
    use_contextual = fields.get("use_contextual", "false") == "true"
    use_late_chunking = fields.get("use_late_chunking", "false") == "true"
    try:
        _run_ingest_job(job_id, url, use_contextual, use_late_chunking)
    finally:
        try:
            r.xack(_STREAM_KEY, _STREAM_GROUP, msg_id)
        except Exception as e:
            logger.warning(f"[StreamWorker] xack failed {msg_id}: {e}")


def _stream_worker_loop() -> None:
    """Daemon thread: pulls ingest jobs from Redis Stream with consumer group."""
    global _stream_worker_active
    r = _get_redis()
    if not r:
        logger.warning("[StreamWorker] Redis unavailable — worker not started, fallback to threads")
        return

    # Create consumer group — ignore BUSYGROUP if already exists
    try:
        r.xgroup_create(_STREAM_KEY, _STREAM_GROUP, id="0", mkstream=True)
        logger.info(f"[StreamWorker] Consumer group '{_STREAM_GROUP}' created")
    except Exception:
        pass

    # Crash recovery: reclaim messages stuck > 60s from a previous process
    try:
        pending = r.xpending_range(_STREAM_KEY, _STREAM_GROUP, "-", "+", count=20)
        for p in pending:
            if p["time_since_delivered"] > 60_000:
                claimed = r.xclaim(_STREAM_KEY, _STREAM_GROUP, _STREAM_CONSUMER, 60_000, [p["message_id"]])
                for claim_id, fields in claimed:
                    logger.info(f"[StreamWorker] Recovering stuck job {fields.get('job_id')}")
                    _process_stream_msg(r, claim_id, fields)
    except Exception as e:
        logger.warning(f"[StreamWorker] Pending recovery error: {e}")

    _stream_worker_active = True
    logger.info("[StreamWorker] Ready — listening on 'yourag:ingest'")

    while _stream_worker_active:
        try:
            messages = r.xreadgroup(
                _STREAM_GROUP, _STREAM_CONSUMER,
                {_STREAM_KEY: ">"},
                count=1,
                block=5000,  # 5s timeout so we can check _stream_worker_active
            )
            if not messages:
                continue
            for _sname, msgs in messages:
                for msg_id, fields in msgs:
                    logger.info(f"[StreamWorker] Dequeued job {fields.get('job_id')}")
                    _process_stream_msg(r, msg_id, fields)
        except Exception as e:
            if _stream_worker_active:
                logger.error(f"[StreamWorker] Error: {e}")
                time.sleep(1)


def _invalidate_bm25_cache(collection_name: str) -> None:
    """Invalidate BM25 in-memory cache for a collection (called on ingest and delete)."""
    if AIStore.hybrid_retriever:
        AIStore.hybrid_retriever.sparse._bm25_cache.pop(collection_name, None)
        AIStore.hybrid_retriever.sparse._doc_mappings.pop(collection_name, None)


def _run_ingest_job(job_id: str, url: str, use_contextual: bool, use_late_chunking: bool) -> None:
    """Background task: chạy ingest và cập nhật trạng thái job."""
    _job_update(job_id, {"status": "running"})
    try:
        pipeline = IngestionPipeline(
            use_contextual_enrichment=use_contextual,
            use_late_chunking=use_late_chunking,
        )
        result = pipeline.run(url)
        # Invalidate BM25 cache so next query uses fresh chunks
        if result.get("collection_name"):
            _invalidate_bm25_cache(result["collection_name"])
        _job_update(job_id, {"status": "done", "result": result})
    except Exception as e:
        logger.error(f"[IngestJob {job_id}] failed: {e}")
        _job_update(job_id, {"status": "error", "error": str(e)})

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
        # Invalidate in-memory graph cache
        if AIStore.graph_retriever:
            AIStore.graph_retriever._graph_cache.pop(collection_name, None)
        # Invalidate Redis graph
        from src.core.redis_client import get_redis as _get_redis_client
        _r = _get_redis_client()
        if _r:
            _r.delete(f"graph:{collection_name}")
        # Invalidate BM25 cache trong HybridRetriever singleton
        _invalidate_bm25_cache(collection_name)
        logger.info(f"🗑️ Deleted collection: {collection_name}")
        return {"status": "deleted", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
@limiter.limit("5/minute")
async def ingest_video(request: Request, req: IngestRequest, background_tasks: BackgroundTasks, _: str = Depends(require_api_key)):
    """Khởi động ingest video bất đồng bộ. Trả về job_id để theo dõi tiến độ."""
    if not _YOUTUBE_URL_RE.match(req.url.strip()):
        raise HTTPException(status_code=422, detail="Invalid YouTube URL. Supported formats: youtube.com/watch?v=..., youtu.be/..., youtube.com/shorts/...")
    job_id = str(uuid.uuid4())
    _job_set(job_id, {"status": "queued", "url": req.url})

    r = _get_redis()
    if r and _stream_worker_active:
        # Redis Streams path — survives API restart, enables crash recovery
        r.xadd(_STREAM_KEY, {
            "job_id": job_id,
            "url": req.url,
            "use_contextual": str(req.use_contextual).lower(),
            "use_late_chunking": str(req.use_late_chunking).lower(),
        })
        logger.info(f"[Ingest] Job {job_id} queued via Redis Stream")
    else:
        # Fallback — Redis unavailable or worker not ready
        background_tasks.add_task(
            _run_ingest_job, job_id, req.url, req.use_contextual, req.use_late_chunking
        )
        logger.info(f"[Ingest] Job {job_id} queued via BackgroundTasks (Redis fallback)")

    return {"job_id": job_id, "status": "queued"}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    """Kiểm tra trạng thái job ingest: queued → running → done / error."""
    job = _job_get(job_id)
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
@limiter.limit("20/minute")
async def chat_rag(request: Request, req: ChatRequest, _: str = Depends(require_api_key)):
    try:
        session_id = req.session_id or str(uuid.uuid4())
        history_mgr = ChatHistoryManager(session_id=session_id, collection_name=req.collection)
        history_mgr.add_message(role="user", content=req.query)
        chat_history_str = history_mgr.format_for_prompt()

        # --- CHECK SEMANTIC CACHE ---
        cached_data = AIStore.cache.check_cache(req.query, collection_name=req.collection)
        if cached_data:
            answer = cached_data["answer"]
            sources = cached_data["sources"]
            facts = cached_data["facts"]

            history_mgr.add_message(role="assistant", content=answer)
            return {
                "answer": answer,
                "sources": sources,
                "facts": facts,
                "graph_entities": [],
                "session_id": session_id,
                "cached": True
            }

        # 1. Hybrid Search
        hybrid = AIStore.hybrid_retriever or HybridRetriever(top_k=10)
        candidates = hybrid.search(req.query, collection_name=req.collection)

        if not candidates:
            return {"answer": "Không tìm thấy thông tin phù hợp trong video này.", "sources": [], "facts": [], "session_id": session_id}

        # 2. Graph RAG Search (lấy graph_data thật)
        graph_data = AIStore.graph_retriever.search(req.query, collection_name=req.collection)

        # 3. Rerank
        final_chunks = AIStore.reranker.rerank(query=req.query, chunks=candidates, top_k=9)

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
            collection_name=req.collection,
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
@limiter.limit("20/minute")
async def chat_rag_stream(request: Request, req: ChatRequest, _: str = Depends(require_api_key)):
    """Endpoint trả về Streaming Response với Global Context."""
    try:
        session_id = req.session_id or str(uuid.uuid4())
        history_mgr = ChatHistoryManager(session_id=session_id, collection_name=req.collection)
        history_mgr.add_message(role="user", content=req.query)
        chat_history_str = history_mgr.format_for_prompt()

        # --- CHECK SEMANTIC CACHE ---
        cached_data = AIStore.cache.check_cache(req.query, collection_name=req.collection)
        if cached_data:
            def cached_generator():
                answer = cached_data["answer"]
                yield answer
                history_mgr.add_message(role="assistant", content=answer)
                meta = json.dumps({
                    "sources": cached_data.get("sources", []),
                    "facts": cached_data.get("facts", []),
                    "session_id": session_id,
                    "suggestions": [],
                    "cached": True,
                })
                yield f"\n\n__META__{meta}"

            return StreamingResponse(cached_generator(), media_type="text/plain")

        # 1. Hybrid Search
        hybrid = AIStore.hybrid_retriever or HybridRetriever(top_k=10)
        candidates = hybrid.search(req.query, collection_name=req.collection)

        if not candidates:
            async def no_result():
                yield "Không tìm thấy thông tin phù hợp trong video này."
            return StreamingResponse(no_result(), media_type="text/plain")

        # 2. Graph RAG Search (lấy graph_data thật)
        graph_data = AIStore.graph_retriever.search(req.query, collection_name=req.collection)

        # 3. Rerank
        final_chunks = AIStore.reranker.rerank(query=req.query, chunks=candidates, top_k=9)

        # 4. Global Summary
        global_summary = AIStore.summarizer.summarize(req.collection)

        # 5. Streaming Generator
        def response_generator():
            full_response = ""
            _lf = get_langfuse()
            _trace = None
            _gen_span = None

            if _lf:
                try:
                    _trace = _lf.trace(
                        name="rag-query",
                        session_id=session_id,
                        input={"query": req.query, "collection": req.collection},
                        tags=["streaming"],
                    )
                    # Retrieval is already complete — log as a finished span
                    _retrieval_span = _trace.span(
                        name="retrieval",
                        input={"query": req.query, "top_k": 10},
                    )
                    _retrieval_span.end(output={
                        "candidates": len(candidates),
                        "reranked": len(final_chunks),
                        "top_score": round(final_chunks[0].get("hybrid_score", 0), 4) if final_chunks else 0,
                        "graph_facts": len(graph_data.get("facts", [])),
                    })
                    _gen_span = _trace.generation(
                        name="llm-stream",
                        model=settings.LLM_MODEL_NAME,
                        model_parameters={"temperature": 0.2, "max_tokens": 2000},
                        input={
                            "chunks_count": len(final_chunks),
                            "has_graph": bool(graph_data.get("facts")),
                        },
                    )
                except Exception:
                    pass

            try:
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
            except Exception as e:
                logger.error(f"[Stream] Generation error: {e}")
                if _lf:
                    try:
                        if _gen_span:
                            _gen_span.end(level="ERROR", status_message=str(e))
                        if _trace:
                            _trace.update(level="ERROR", status_message=str(e))
                        _lf.flush()
                    except Exception:
                        pass
                yield f"\n\n[Error: {str(e)}]"
                return

            if _lf:
                try:
                    if _gen_span:
                        _gen_span.end(
                            output=full_response[:2000],
                            usage={"output": len(full_response.split())},
                        )
                    if _trace:
                        _trace.update(output={"answer_length": len(full_response)})
                except Exception:
                    pass

            # Lưu lại tin nhắn AI sau khi stream xong
            history_mgr.add_message(role="assistant", content=full_response)

            from src.core.utils import format_timestamp
            sources = [
                f"{format_timestamp(c['metadata']['start_time'])}–{format_timestamp(c['metadata']['end_time'])}"
                for c in final_chunks
            ]

            # --- SAVE TO CACHE ---
            AIStore.cache.save_to_cache(
                query=req.query,
                answer=full_response,
                collection_name=req.collection,
                sources=sources,
                facts=graph_data.get("facts", [])
            )

            # Generate Suggested Questions
            suggestions = []
            sq_prompt = f"Dựa trên câu hỏi '{req.query}' và câu trả lời '{full_response}', hãy gợi ý đúng 3 câu hỏi ngắn gọn (mỗi câu dưới 12 từ) mà người dùng có thể hỏi tiếp theo. Trả về định dạng: Câu 1|Câu 2|Câu 3. Không gạch đầu dòng, không đánh số."
            try:
                sq_list = AIStore.generator.llm.chat_complete(sq_prompt, system="Bạn là trợ lý RAG.", max_tokens=100)
                suggestions = [s.strip() for s in sq_list.replace('\n', '').split("|") if s.strip()]
            except Exception as e:
                logger.error(f"Error generating suggestions: {e}")

            if _lf:
                try:
                    _lf.flush()
                except Exception:
                    pass

            # Single JSON meta frame — safe against LLM output containing delimiter strings
            meta = json.dumps({
                "sources": sources,
                "facts": graph_data.get("facts", []),
                "session_id": session_id,
                "suggestions": suggestions[:3],
                "cached": False,
            })
            yield f"\n\n__META__{meta}"

        return StreamingResponse(response_generator(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggestions/{collection}")
@limiter.limit("10/minute")
async def get_suggestions(request: Request, collection: str, _: str = Depends(require_api_key)):
    """Generate 4 câu hỏi gợi ý dựa trên nội dung video."""
    try:
        summary = AIStore.summarizer.summarize(collection)
        prompt = (
            f"Dựa trên tóm tắt video sau:\n\n{summary}\n\n"
            "Hãy gợi ý đúng 4 câu hỏi ngắn gọn (mỗi câu dưới 10 từ) mà người xem có thể hỏi về video này. "
            "Trả về định dạng: Câu 1|Câu 2|Câu 3|Câu 4. Không gạch đầu dòng, không đánh số, không giải thích thêm."
        )
        raw = AIStore.generator.llm.chat_complete(prompt, system="Bạn là trợ lý RAG.", max_tokens=120)
        suggestions = [s.strip() for s in raw.replace("\n", "").split("|") if s.strip()]
        return {"suggestions": suggestions[:4]}
    except Exception as e:
        logger.error(f"[suggestions] {e}")
        return {"suggestions": []}


@app.get("/summarize/{collection}")
@limiter.limit("10/minute")
async def get_summary(request: Request, collection: str, _: str = Depends(require_api_key)):
    try:
        summary = AIStore.summarizer.summarize(collection)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summarize/stream/{collection}")
@limiter.limit("10/minute")
async def get_summary_stream(request: Request, collection: str, _: str = Depends(require_api_key)):
    """Endpoint tóm tắt video theo kiểu Streaming."""
    try:
        return StreamingResponse(
            AIStore.summarizer.summarize_stream(collection),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
