from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class IngestRequest(BaseModel):
    video_url: HttpUrl
    chunk_size: int = 500
    overlap: int = 50

class Citation(BaseModel):
    video_url: str
    timestamp_start: str
    text_content: str
    relevance_score: float

class QueryRequest(BaseModel):
    query: str
    force_web_search: bool = False
    filters: Optional[dict] = None # Filter metadata nếu cần

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: float
    cache_hit: bool
