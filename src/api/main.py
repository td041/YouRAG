from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict as TypingDict, List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
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
from src.engine.retrieval.contextual_compressor import ContextualCompressor
from src.api.auth import require_api_key
from src.core.langfuse_client import get_langfuse
import re
import json

_YOUTUBE_URL_RE = re.compile(
    r"^(https?://)?(www\.)?"
    r"(youtube\.com/(watch\?.*v=|shorts/|embed/)|youtu\.be/)"
    r"[A-Za-z0-9_\-]{11}"
)


def _parse_suggestions(raw: str) -> list:
    """Split LLM output into individual suggestion strings.

    Handles all common formats the model returns:
      - "Q1|Q2|Q3"
      - "Q1\\nQ2\\nQ3"
      - "Q1?Q2?Q3?" (no separator, questions joined by their own mark)
    Picks whichever split yields the most non-empty parts.
    """
    def _clean(s: str) -> str:
        s = re.sub(r"^[Qq]\d+[\.\:\s]*", "", s).strip()
        s = re.sub(r"[\s]*[Qq]\d+$", "", s).strip()
        s = re.sub(r"^[\d\.\-\*\s]+", "", s).strip()
        return s

    def _split(text: str, sep: str) -> list:
        if sep == "?":
            return [p.strip() + "?" for p in text.split("?") if p.strip()]
        return [p.strip() for p in text.split(sep) if p.strip()]

    best: list = [_clean(raw.strip())] if _clean(raw.strip()) else []
    for sep in ("|", "\n", "?"):
        candidate = [_clean(s) for s in _split(raw, sep) if _clean(s)]
        if len(candidate) > len(best):
            best = candidate
    return best

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

_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
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
    hybrid_retriever = None
    cache = None
    compressor = None

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
    _load_component("ContextualCompressor", lambda: ContextualCompressor(), "compressor")

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
    use_late_chunking: bool = True   # Jina Late Chunking (cần JINA_API_KEY, fallback bge-m3 nếu lỗi)
    use_visual_rag: bool = False     # Visual Frame RAG (cần GEMINI_API_KEY hoặc OPENAI_API_KEY)

class ChatRequest(BaseModel):
    query: str
    collection: Optional[str] = None        # single-video (backward compat)
    collections: Optional[List[str]] = None  # multi-video
    session_id: Optional[str] = None

    @property
    def resolved_collections(self) -> List[str]:
        if self.collections:
            return self.collections
        if self.collection:
            return [self.collection]
        raise ValueError("Phải truyền 'collection' hoặc 'collections'")

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
    use_visual_rag = fields.get("use_visual_rag", "false") == "true"
    try:
        _run_ingest_job(job_id, url, use_contextual, use_late_chunking, use_visual_rag)
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


def _invalidate_splade_cache(collection_name: str) -> None:
    """Invalidate SPLADE in-memory vector cache for a collection (called on ingest and delete)."""
    if AIStore.hybrid_retriever:
        AIStore.hybrid_retriever.sparse.clear_cache(collection_name)


def _pregen_summary(collection_name: str) -> None:
    """Pre-generate và cache summary vào Redis ngay sau ingest.
    Chạy trong background thread để không block ingest job.
    Khi user hỏi lần đầu, summary đã sẵn trong cache.
    """
    try:
        if AIStore.summarizer:
            logger.info(f"[PreGen] Generating summary for '{collection_name}'...")
            AIStore.summarizer.summarize(collection_name)
            logger.info(f"[PreGen] Summary cached for '{collection_name}'")
    except Exception as e:
        logger.warning(f"[PreGen] Summary pre-gen failed for '{collection_name}': {e}")


def _run_ingest_job(
    job_id: str, url: str, use_contextual: bool, use_late_chunking: bool, use_visual_rag: bool = False
) -> None:
    """Background task: chạy ingest và cập nhật trạng thái job."""
    _job_update(job_id, {"status": "running"})
    try:
        pipeline = IngestionPipeline(
            use_contextual_enrichment=use_contextual,
            use_late_chunking=use_late_chunking,
            use_visual_rag=use_visual_rag,
        )
        result = pipeline.run(url)
        collection_name = result.get("collection_name")
        if collection_name:
            _invalidate_splade_cache(collection_name)
            # Pre-generate summary vào Redis ngay sau ingest
            # → Q&A đầu tiên sẽ có context tổng quan, không cần gọi LLM trong chat
            t = threading.Thread(
                target=_pregen_summary, args=(collection_name,), daemon=True,
                name=f"pregen-summary-{collection_name[:20]}"
            )
            t.start()
        _job_update(job_id, {"status": "done", "result": result})
    except Exception as e:
        logger.error(f"[IngestJob {job_id}] failed: {e}")
        _job_update(job_id, {"status": "error", "error": str(e)})

@app.get("/collections")
def list_collections():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    collections_response = db_instance.client.get_collections()
    names = [c.name for c in collections_response.collections if c.name not in _INTERNAL_COLLECTIONS]

    def _fetch_meta(name: str) -> dict:
        try:
            records, _ = db_instance.client.scroll(
                collection_name=name, limit=1,
                with_payload=True, with_vectors=False
            )
            if records and records[0].payload:
                meta = records[0].payload
                return {"name": name, "title": meta.get("title", name), "video_id": meta.get("video_id")}
        except Exception:
            pass
        return {"name": name, "title": name, "video_id": None}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_meta, n): n for n in names}
        results = {futures[f]: f.result() for f in as_completed(futures)}

    return [results[n] for n in names if n in results]

@app.get("/benchmark/report")
def get_benchmark_report():
    """Trả về kết quả RAGAS benchmark gần nhất từ tests/benchmark/ragas_report.json."""
    import os
    import json as _json
    import math

    def _sanitize(obj):
        """Đệ quy thay NaN/Inf bằng None để JSON chuẩn chấp nhận."""
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    report_path = os.path.join("tests", "benchmark", "ragas_report.json")
    if not os.path.exists(report_path):
        return {"available": False}
    try:
        with open(report_path, encoding="utf-8") as f:
            # parse_constant=None compat: dùng raw text decode để giữ NaN
            raw = f.read()
        data = _json.loads(raw.replace(": NaN", ": null").replace(":NaN", ":null"))
        return _sanitize({"available": True, **data})
    except Exception as e:
        logger.warning(f"[benchmark] Cannot read report: {e}")
        return {"available": False}

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str, _: str = Depends(require_api_key)):
    """Xóa một collection (video) khỏi Qdrant và graph store."""
    try:
        db_instance.client.delete_collection(collection_name)
        # Xóa knowledge graph (.json — graph_rag.py dùng JSON, không dùng pickle)
        import os
        graph_path = os.path.join("graph_store", f"{collection_name}.json")
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
        # Invalidate SPLADE cache trong HybridRetriever singleton
        _invalidate_splade_cache(collection_name)
        # Invalidate summary cache
        if _r:
            _r.delete(f"summary:{collection_name}")
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
            "use_visual_rag": str(req.use_visual_rag).lower(),
        })
        logger.info(f"[Ingest] Job {job_id} queued via Redis Stream")
    else:
        # Fallback — Redis unavailable or worker not ready
        background_tasks.add_task(
            _run_ingest_job, job_id, req.url, req.use_contextual, req.use_late_chunking, req.use_visual_rag
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
        if AIStore.graph_retriever:
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
        loop = asyncio.get_running_loop()
        session_id = req.session_id or str(uuid.uuid4())
        history_mgr = ChatHistoryManager(session_id=session_id, collection_name=req.collection)
        history_mgr.add_message(role="user", content=req.query)
        chat_history_str = history_mgr.format_for_prompt()

        # --- CHECK SEMANTIC CACHE ---
        cached_data = AIStore.cache.check_cache(req.query, collection_name=req.collection) if AIStore.cache else None
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

        # 1. Hybrid Search (blocking → offload)
        hybrid = AIStore.hybrid_retriever or HybridRetriever(top_k=10)
        candidates = await loop.run_in_executor(
            None, lambda: hybrid.search(req.query, collection_name=req.collection)
        )

        if not candidates:
            return {"answer": "Không tìm thấy thông tin phù hợp trong video này.", "sources": [], "facts": [], "session_id": session_id}

        # 2. Graph RAG Search (lấy graph_data thật)
        graph_data = (
            AIStore.graph_retriever.search(req.query, collection_name=req.collection)
            if AIStore.graph_retriever else {"facts": [], "graph_summary": ""}
        )

        # 3. Rerank (blocking → offload)
        final_chunks = await loop.run_in_executor(
            None, lambda: AIStore.reranker.rerank(query=req.query, chunks=candidates, top_k=9)
        )

        # 4. Contextual Compression — bỏ câu noise trước khi pass LLM (blocking → offload)
        if AIStore.compressor:
            try:
                _chunks = final_chunks
                final_chunks = await loop.run_in_executor(
                    None, lambda: AIStore.compressor.compress(req.query, _chunks)
                )
            except Exception as e:
                logger.warning(f"[Compressor] Failed in /chat: {e}")

        # 5. Global Summary — chỉ đọc Redis cache, không generate mới
        global_summary = ""
        try:
            from src.engine.generation.summarizer import _get_redis as _sum_redis
            _r = _sum_redis()
            if _r:
                _cached = _r.get(f"summary:{req.collection}")
                if _cached:
                    global_summary = _cached
        except Exception:
            pass

        # 6. Generate Answer (blocking → offload)
        _fc = final_chunks
        answer = await loop.run_in_executor(
            None, lambda: AIStore.generator.generate(
                query=req.query,
                retrieved_chunks=_fc,
                global_summary=global_summary,
                graph_facts=graph_data.get("facts", []),
                graph_summary=graph_data.get("graph_summary", ""),
                chat_history=chat_history_str,
            )
        )

        history_mgr.add_message(role="assistant", content=answer)

        from src.core.utils import format_timestamp
        sources = [
            f"{format_timestamp(c.get('metadata', {}).get('start_time', 0.0))}–{format_timestamp(c.get('metadata', {}).get('end_time', 0.0))}"
            for c in final_chunks
        ]

        # --- SAVE TO CACHE ---
        if AIStore.cache:
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
    # Session setup — lightweight, no I/O
    try:
        session_id = req.session_id or str(uuid.uuid4())
        collections_list = req.resolved_collections
        history_mgr = ChatHistoryManager(session_id=session_id, collection_name=collections_list[0])
        history_mgr.add_message(role="user", content=req.query)
        chat_history_str = history_mgr.format_for_prompt()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def stream():
        """Async generator: stream progress updates + final LLM response.
        All blocking CPU/IO work is offloaded via run_in_executor so the
        event loop stays responsive. Progress lines (prefix __PROGRESS__)
        are stripped by the frontend and shown as status indicators.
        """
        loop = asyncio.get_running_loop()
        t_start = time.perf_counter()

        def _ms(t0: float, t1: float) -> int:
            return round((t1 - t0) * 1000)

        try:
            # --- CHECK SEMANTIC CACHE (fast, no blocking) ---
            t0 = time.perf_counter()
            cached_data = (
                AIStore.cache.check_cache(req.query, collection_name=collections_list[0])
                if AIStore.cache else None
            )
            t_cache_ms = _ms(t0, time.perf_counter())
            if cached_data:
                answer = AIStore.generator._strip_superscripts(cached_data["answer"])
                yield answer
                history_mgr.add_message(role="assistant", content=answer)
                logger.info(f"[Latency] cache_hit=True cache={t_cache_ms}ms query='{req.query[:60]}'")
                meta = json.dumps({
                    "sources": cached_data.get("sources", []),
                    "facts": cached_data.get("facts", []),
                    "session_id": session_id,
                    "suggestions": [],
                    "cached": True,
                    "latency_ms": {"cache": t_cache_ms, "total": t_cache_ms},
                })
                yield f"\n\n__META__{meta}"
                return

            # 1. Hybrid Search (blocking: SPLADE index + ThreadPoolExecutor)
            yield "__PROGRESS__Đang tìm kiếm tài liệu...\n"
            hybrid = AIStore.hybrid_retriever or HybridRetriever(top_k=10)
            t0 = time.perf_counter()
            try:
                candidates = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: hybrid.search_multi(req.query, collections_list)),
                    timeout=45.0,
                )
            except asyncio.TimeoutError:
                logger.error("[Stream] Hybrid search timed out after 45s")
                yield "Hệ thống quá tải, vui lòng thử lại sau."
                return
            t_search_ms = _ms(t0, time.perf_counter())

            if not candidates:
                yield "Không tìm thấy thông tin phù hợp trong video này."
                return

            # 2. Graph RAG Search — fast (reads from Redis cache)
            t0 = time.perf_counter()
            all_facts: list = []
            all_graph_summary = ""
            for _col in collections_list:
                try:
                    _gd = AIStore.graph_retriever.search(req.query, collection_name=_col)
                    all_facts.extend(_gd.get("facts", []))
                    if _gd.get("graph_summary"):
                        all_graph_summary = _gd["graph_summary"]
                except Exception:
                    pass
            graph_data = {"facts": all_facts, "graph_summary": all_graph_summary}
            t_graph_ms = _ms(t0, time.perf_counter())

            # 3. Rerank (blocking: cross-encoder on GPU)
            yield "__PROGRESS__Đang reranking kết quả...\n"
            t0 = time.perf_counter()
            try:
                final_chunks = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: AIStore.reranker.rerank(query=req.query, chunks=candidates, top_k=9),
                    ),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("[Stream] Reranker timed out, using raw candidates")
                final_chunks = candidates[:9]
            t_rerank_ms = _ms(t0, time.perf_counter())

            # 4. Contextual Compression
            yield "__PROGRESS__Đang nén ngữ cảnh...\n"
            _fc = final_chunks
            _col0 = collections_list[0]
            t0 = time.perf_counter()
            try:
                final_chunks = await loop.run_in_executor(
                    None,
                    lambda: AIStore.compressor.compress(req.query, _fc) if AIStore.compressor else _fc,
                )
            except Exception as e:
                logger.warning(f"[Compressor] Failed, using raw chunks: {e}")
            t_compress_ms = _ms(t0, time.perf_counter())

            # 5. Summary — CHỈ lấy từ Redis cache, không generate mới trong chat.
            # Background pre-generation đã chạy sau ingest — nếu chưa có cache thì skip.
            t0 = time.perf_counter()
            global_summary = ""
            try:
                from src.engine.generation.summarizer import _get_redis as _sum_redis
                _r = _sum_redis()
                if _r:
                    _cached = _r.get(f"summary:{_col0}")
                    if _cached:
                        global_summary = _cached
                        logger.info(f"[Chat] Summary cache hit for [{_col0}]")
                    else:
                        logger.info("[Chat] Summary cache miss — skipping LLM call in stream")
            except Exception:
                pass
            t_summary_ms = _ms(t0, time.perf_counter())

            t_ttft_ms = _ms(t_start, time.perf_counter())  # snapshot before first LLM token

            # 6. Langfuse tracing setup
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
                    _retrieval_span = _trace.span(
                        name="retrieval", input={"query": req.query, "top_k": 10}
                    )
                    _retrieval_span.end(output={
                        "candidates": len(candidates),
                        "reranked": len(final_chunks),
                        "top_score": round(final_chunks[0].get("hybrid_score", 0), 4) if final_chunks else 0,
                        "graph_facts": len(graph_data.get("facts", [])),
                        "latency_ms": {
                            "cache": t_cache_ms,
                            "search": t_search_ms,
                            "graph": t_graph_ms,
                            "rerank": t_rerank_ms,
                            "compress": t_compress_ms,
                            "summary": t_summary_ms,
                            "ttft": t_ttft_ms,
                        },
                    })
                    _gen_span = _trace.generation(
                        name="llm-stream",
                        model=settings.LLM_MODEL_NAME,
                        model_parameters={"temperature": 0.2, "max_tokens": 2000},
                        input={"chunks_count": len(final_chunks), "has_graph": bool(graph_data.get("facts"))},
                    )
                except Exception:
                    pass

            # 7. LLM generation — buffer server-side so citations can be validated
            #    before any text reaches the client, then re-stream the clean response.
            yield "__PROGRESS__Đang tạo câu trả lời...\n"
            full_response = ""
            t0 = time.perf_counter()
            first_token = True
            t_first_token_ms = 0
            try:
                for chunk in AIStore.generator.generate_stream(
                    query=req.query,
                    retrieved_chunks=final_chunks,
                    global_summary=global_summary,
                    graph_facts=graph_data.get("facts", []),
                    graph_summary=graph_data.get("graph_summary", ""),
                    chat_history=chat_history_str,
                ):
                    if first_token:
                        t_first_token_ms = _ms(t0, time.perf_counter())
                        first_token = False
                    full_response += chunk

                # Validate & clean before sending to client
                full_response = AIStore.generator._strip_superscripts(full_response)
                full_response = AIStore.generator._validate_citations(full_response, final_chunks)

                # Re-stream validated text in small chunks for UI responsiveness
                _chunk_size = 40
                for _i in range(0, len(full_response), _chunk_size):
                    yield full_response[_i:_i + _chunk_size]
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
                history_mgr.add_message(role="assistant", content=f"[Lỗi: {e}]")
                yield f"\n\n[Lỗi: {str(e)}]"
                return

            t_total_ms = _ms(t_start, time.perf_counter())
            t_llm_ms = _ms(t0, time.perf_counter())
            logger.info(
                f"[Latency] query='{req.query[:60]}' "
                f"cache={t_cache_ms}ms search={t_search_ms}ms graph={t_graph_ms}ms "
                f"rerank={t_rerank_ms}ms compress={t_compress_ms}ms summary={t_summary_ms}ms "
                f"ttft={t_ttft_ms}ms llm_first_token={t_first_token_ms}ms llm_total={t_llm_ms}ms "
                f"TOTAL={t_total_ms}ms candidates={len(candidates)} chunks={len(final_chunks)}"
            )

            if _lf:
                try:
                    if _gen_span:
                        _gen_span.end(
                            output=full_response[:2000],
                            usage={"output": len(full_response.split())},
                        )
                    if _trace:
                        _trace.update(output={
                            "answer_length": len(full_response),
                            "latency_ms": {
                                "cache": t_cache_ms,
                                "search": t_search_ms,
                                "graph": t_graph_ms,
                                "rerank": t_rerank_ms,
                                "compress": t_compress_ms,
                                "summary": t_summary_ms,
                                "ttft": t_ttft_ms,
                                "llm_first_token": t_first_token_ms,
                                "llm_total": t_llm_ms,
                                "total": t_total_ms,
                            },
                        })
                except Exception:
                    pass

            history_mgr.add_message(role="assistant", content=full_response)

            from src.core.utils import format_timestamp
            sources = [
                {
                    "label": f"{format_timestamp(c['metadata']['start_time'])}–{format_timestamp(c['metadata']['end_time'])}",
                    "start_time": c["metadata"].get("start_time", 0),
                    "video_id": c["metadata"].get("video_id"),
                    "title": c["metadata"].get("title"),
                    "chunk_type": c["metadata"].get("chunk_type", "text"),
                }
                for c in final_chunks
            ]

            if AIStore.cache:
                AIStore.cache.save_to_cache(
                    query=req.query,
                    answer=full_response,
                    collection_name=collections_list[0],
                    sources=sources,
                    facts=graph_data.get("facts", []),
                )

            suggestions = []
            video_excerpts = "\n".join(f"- {c['content'][:150]}" for c in final_chunks[:4])
            sq_prompt = (
                f"These are real excerpts from the video:\n{video_excerpts}\n\n"
                f"The user just asked: '{req.query}'\n"
                "Suggest exactly 3 follow-up questions that can ONLY be answered "
                "from the video excerpts above — not from general knowledge. "
                "Output ONLY: question1|question2|question3"
            )
            try:
                sq_list = AIStore.generator.llm.chat_complete(
                    sq_prompt, system="You are a helpful assistant.", max_tokens=80
                )
                suggestions = _parse_suggestions(sq_list)
            except Exception as e:
                logger.error(f"Error generating suggestions: {e}")

            if _lf:
                try:
                    _lf.flush()
                except Exception:
                    pass

            meta = json.dumps({
                "sources": sources,
                "facts": graph_data.get("facts", []),
                "session_id": session_id,
                "suggestions": suggestions[:3],
                "cached": False,
                "latency_ms": {
                    "search": t_search_ms,
                    "graph": t_graph_ms,
                    "rerank": t_rerank_ms,
                    "ttft": t_ttft_ms,
                    "llm": t_llm_ms,
                    "total": t_total_ms,
                },
            })
            yield f"\n\n__META__{meta}"

        except Exception as e:
            logger.error(f"[Stream] Unexpected error: {e}", exc_info=True)
            yield f"\n\n[Lỗi hệ thống: {str(e)}]"

    return StreamingResponse(stream(), media_type="text/plain")

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
        return {"suggestions": _parse_suggestions(raw)[:4]}
    except Exception as e:
        logger.error(f"[suggestions] {e}")
        return {"suggestions": []}


@app.get("/quiz/{collection}")
@limiter.limit("10/minute")
async def generate_quiz(
    request: Request,
    collection: str,
    count: int = 5,
    mode: str = "quiz",
    _: str = Depends(require_api_key),
):
    """Generate quiz questions or flashcards from a video collection."""
    import json as _json
    try:
        records, _ = db_instance.client.scroll(
            collection_name=collection,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            raise HTTPException(status_code=404, detail="Collection not found or empty")

        import re as _re  # noqa: PLC0415
        all_chunks = sorted(
            [{"text": r.payload.get("text", ""), "start_time": r.payload.get("start_time", 0.0)}
             for r in records if r.payload and r.payload.get("text")],
            key=lambda x: x["start_time"],
        )

        # Ưu tiên chunks có nhiều nội dung thực chất (số liệu, tên cụ thể, thông số)
        def chunk_score(c: dict) -> float:
            t = c["text"]
            score = len(t)  # base: length
            score += len(_re.findall(r'\d+', t)) * 30          # bonus: numbers
            score += len(_re.findall(r'[A-Z]{2,}', t)) * 20   # bonus: acronyms/model names
            score -= 50 if len(t) < 80 else 0                 # penalty: too short
            return score

        chunks = sorted(all_chunks, key=chunk_score, reverse=True)

        MAX_CHARS = 400
        MAX_CHUNKS = 25
        # Take top scored chunks, then re-sort by time for coherent transcript
        sampled = sorted(chunks[:MAX_CHUNKS], key=lambda x: x["start_time"])

        from src.core.utils import format_timestamp
        transcript = "\n".join(
            f"[{format_timestamp(c['start_time'])}] {c['text'][:MAX_CHARS]}"
            for c in sampled
        )

        count = max(3, min(count, 10))

        if mode == "flashcard":
            prompt = f"""Dựa trên transcript video sau, tạo tối đa {count} flashcard học tập.

QUAN TRỌNG:
- Chỉ tạo flashcard từ những đoạn có THÔNG TIN CỤ THỂ: số liệu, thông số kỹ thuật, tên sản phẩm, so sánh rõ ràng.
- KHÔNG tạo flashcard từ câu chuyện phiếm, cảm nhận mơ hồ, hay câu giới thiệu.
- Nếu một đoạn transcript không có fact cụ thể → bỏ qua, đừng cố tạo card từ đó.
- Trả về ÍT card hơn {count} nếu không đủ nội dung chất lượng. Đừng bịa.

<transcript>
{transcript}
</transcript>

Trả về JSON hợp lệ (không có markdown):
{{
  "cards": [
    {{
      "front": "Câu hỏi về một fact cụ thể (tên, số, thông số...)",
      "back": "Câu trả lời rõ ràng, chính xác, trích từ transcript",
      "timestamp": "X:XX"
    }}
  ]
}}

QUAN TRỌNG VỀ timestamp: Dùng ĐÚNG timestamp [X:XX] từ dòng transcript chứa thông tin đó. KHÔNG dùng "0:00" hay "0:01" mặc định.
Ví dụ front tốt: "Pin Nova thường dung lượng bao nhiêu?" → back: "4000mAh", timestamp: "2:16"
Ví dụ front xấu (KHÔNG làm): "Bàn phím này trông như thế nào?" → back mơ hồ"""
        else:
            prompt = f"""Dựa trên transcript video sau, tạo {count} câu hỏi trắc nghiệm.

QUAN TRỌNG: Chỉ hỏi về thông tin CÓ TRONG transcript. Đáp án đúng phải được nêu rõ trong transcript. Không bịa đặt con số hay sự kiện.

<transcript>
{transcript}
</transcript>

Trả về JSON hợp lệ (không có markdown), đúng format:
{{
  "questions": [
    {{
      "question": "Câu hỏi rõ ràng, cụ thể",
      "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
      "correct": 0,
      "explanation": "Trích dẫn nguyên văn từ transcript để chứng minh đáp án đúng",
      "timestamp": "X:XX"
    }}
  ]
}}

QUAN TRỌNG VỀ timestamp: Dùng ĐÚNG timestamp [X:XX] từ dòng transcript chứa thông tin đó. KHÔNG dùng "0:00" hay "0:01" mặc định.
Quy tắc: correct là index 0-3, options đủ 4 lựa chọn, explanation phải trích dẫn nội dung transcript."""

        system_msg = (
            "Bạn là chuyên gia tạo câu hỏi học tập. "
            "Chỉ tạo câu hỏi dựa trên nội dung được cung cấp, không bịa đặt. "
            "Chỉ trả về JSON thuần, KHÔNG markdown, KHÔNG giải thích thêm."
        )

        def _call_llm() -> dict:
            raw = AIStore.generator.llm.chat_complete(
                prompt=prompt, system=system_msg, max_tokens=2000, temperature=0.4,
            )
            raw = raw.strip()
            match = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object in response: {raw[:200]}")
            return _json.loads(match.group(0))

        def _ts_to_seconds(ts: str) -> float:
            """Parse 'mm:ss' or 'h:mm:ss' string to seconds."""
            try:
                parts = [int(p) for p in ts.strip().split(":")]
                if len(parts) == 2:
                    return parts[0] * 60 + parts[1]
                if len(parts) == 3:
                    return parts[0] * 3600 + parts[1] * 60 + parts[2]
            except Exception:
                pass
            return 0.0

        def _inject_start_times(data: dict) -> dict:
            """Compute start_time from timestamp string so LLM can't get it wrong."""
            for item in data.get("cards", []) + data.get("questions", []):
                item["start_time"] = _ts_to_seconds(item.get("timestamp", "0:00"))
            return data

        # Retry up to 2 times if LLM returns malformed JSON
        for attempt in range(2):
            try:
                data = _inject_start_times(_call_llm())
                return {"mode": mode, "collection": collection, **data}
            except (_json.JSONDecodeError, ValueError) as e:
                logger.warning(f"[quiz] attempt {attempt + 1} failed: {e}")

        raise HTTPException(status_code=500, detail="LLM returned invalid JSON — try again")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[quiz] {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
