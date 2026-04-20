from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import sys

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.database import db_instance
from src.engine.ingestion.pipeline import IngestionPipeline
from src.engine.retrieval.hybrid_search import HybridRetriever
from src.engine.ranking.cross_encoder import CrossEncoderReranker
from src.engine.generation.answer_generator import AnswerGenerator
from src.engine.generation.summarizer import VideoSummarizer
from src.core.logger import setup_logger

logger = setup_logger("YouRAG_API")
app = FastAPI(title="YouRAG Backend API", version="1.0.0")

# ─────────────────────────────────────────────
# KHỞI TẠO SINGLETON MODELS (Load sẵn vào RAM/GPU)
# ─────────────────────────────────────────────
class AIStore:
    pipeline = None
    reranker = None
    generator = None
    summarizer = None

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Đang khởi động Backend YouRAG...")
    # Load mồi các mô hình nặng nề tại đây
    AIStore.pipeline = IngestionPipeline()
    AIStore.reranker = CrossEncoderReranker()
    AIStore.generator = AnswerGenerator()
    AIStore.summarizer = VideoSummarizer()
    logger.info("✅ TẤT CẢ MÔ HÌNH ĐÃ SẴN SÀNG TRONG RAM/GPU!")

# ─────────────────────────────────────────────
# MODELS (Pydantic)
# ─────────────────────────────────────────────
class IngestRequest(BaseModel):
    url: str
    use_contextual: bool = False

class ChatRequest(BaseModel):
    query: str
    collection: str

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "online", "message": "YouRAG API is ready."}

@app.get("/collections")
def list_collections():
    # Gọi danh sách Qdrant Collection Objects
    collections_response = db_instance.client.get_collections()
    detailed_collections = []
    
    for c in collections_response.collections:
        try:
            # Lấy 1 record mồi để lấy tiêu đề video (payload)
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

@app.post("/chat")
async def chat_rag(req: ChatRequest):
    try:
        # 1. Hybrid Search (Cân bằng lại 10 chunks để Local chạy nhanh mà vẫn đủ ý)
        hybrid = HybridRetriever(top_k=10)
        candidates, graph_data = hybrid.search(req.query, collection_name=req.collection)
        
        if not candidates:
            return {"answer": "Không tìm thấy thông tin phù hợp trong video này.", "sources": [], "facts": []}

        # 2. Rerank (Dùng mMiniLMv2 siêu tốc)
        final_chunks = AIStore.reranker.rerank(query=req.query, chunks=candidates, top_k=4)

        # 3. Lấy Global Summary để AI không bị "ngáo" khi đếm
        global_summary = AIStore.summarizer.summarize(req.collection)

        # 4. Generate Answer
        answer = AIStore.generator.generate(
            query=req.query, 
            retrieved_chunks=final_chunks,
            global_summary=global_summary,
            graph_facts=graph_data.get("facts", []),
            graph_summary=graph_data.get("graph_summary", "")
        )
        
        # 4. Format Sources
        from src.core.utils import format_timestamp
        sources = [f"{format_timestamp(c['metadata']['start_time'])}–{format_timestamp(c['metadata']['end_time'])}" for c in final_chunks]
        
        return {
            "answer": answer, 
            "sources": sources,
            "facts": graph_data.get("facts", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_rag_stream(req: ChatRequest):
    """Endpoint trả về Streaming Response với Global Context."""
    try:
        # 1. Hybrid Search (Cân bằng 10 chunks)
        hybrid = HybridRetriever(top_k=10)
        candidates, graph_data = hybrid.search(req.query, collection_name=req.collection)
        
        if not candidates:
            # Fake stream báo lỗi
            async def no_result():
                yield "Không tìm thấy thông tin phù hợp trong video này."
            return StreamingResponse(no_result(), media_type="text/plain")

        # 2. Rerank
        final_chunks = AIStore.reranker.rerank(query=req.query, chunks=candidates, top_k=4)

        # 3. Lấy bối cảnh toàn cục (nhanh vì đã cached trong summarizer nếu cần)
        global_summary = AIStore.summarizer.summarize(req.collection)

        # 4. Generate Stream Generator
        def response_generator():
            for chunk in AIStore.generator.generate_stream(
                query=req.query, 
                retrieved_chunks=final_chunks,
                global_summary=global_summary,
                graph_facts=graph_data.get("facts", []),
                graph_summary=graph_data.get("graph_summary", "")
            ):
                # Yield từng text chunk
                yield chunk
            
            # Gửi dòng phân tách và metadata (sources) ở cuối cùng
            from src.core.utils import format_timestamp
            
            sources = [f"{format_timestamp(c['metadata']['start_time'])}–{format_timestamp(c['metadata']['end_time'])}" for c in final_chunks]
            # Đóng gói sources và facts vào chuỗi đặc biệt để frontend bắt được
            yield f"\n\n__SOURCES__::{','.join(sources)}"
            
            if graph_data.get("facts"):
                yield f"\n\n__FACTS__::{'|'.join(graph_data['facts'])}"
            
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
