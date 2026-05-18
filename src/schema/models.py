import re
from pydantic import BaseModel, HttpUrl, field_validator
from typing import List, Optional

_YOUTUBE_RE = re.compile(
    r"(youtube\.com/(watch\?.*v=|shorts/|embed/)|youtu\.be/)"
)


class IngestRequest(BaseModel):
    video_url: HttpUrl
    chunk_size: int = 500
    overlap: int = 50

    @field_validator("video_url")
    @classmethod
    def must_be_youtube(cls, v: HttpUrl) -> HttpUrl:
        if not _YOUTUBE_RE.search(str(v)):
            raise ValueError("URL phải là link YouTube hợp lệ (youtube.com hoặc youtu.be)")
        return v

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
