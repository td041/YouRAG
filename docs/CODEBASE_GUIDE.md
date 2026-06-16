# YouRAG — Codebase Navigation Guide

> Hướng dẫn đọc code, tìm hiểu luồng dữ liệu và mở rộng hệ thống.

---

## Bắt đầu từ đâu?

### Muốn hiểu toàn bộ hệ thống
1. Đọc `ARCHITECTURE.md` — big picture
2. Đọc `src/core/config.py` — mọi config đều ở đây
3. Đọc `src/api/main.py` — entry point của mọi request

### Muốn hiểu ingestion
```
src/engine/ingestion/pipeline.py       ← IngestionPipeline (bắt đầu từ đây)
    → youtube_loader.py                ← fetch transcript + metadata + Whisper STT fallback
    → chunker.py                       ← semantic chunking (pause-aware + vector valleys)
    → contextual_enricher.py           ← optional LLM enrichment
    → late_chunker.py                  ← optional Jina late chunking
    → graph_extractor.py               ← entity extraction → NetworkX
    → src/core/database.py             ← upsert vào Qdrant
```

### Muốn hiểu RAG query
```
src/api/main.py::chat_rag()            ← entry point
    → src/cache/semantic_cache.py      ← check cache trước
    → src/engine/retrieval/hybrid_search.py   ← retrieve
        → dense_search.py              ← Qdrant vector search
        → sparse_search.py             ← BM25 keyword search
    → src/engine/ranking/cross_encoder.py     ← rerank
    → src/engine/generation/answer_generator.py ← generate
        → prompt_builder.py            ← build prompts
        → llm_client.py                ← call Groq API
```

---

## Singletons — Không bao giờ re-instantiate

```python
# Luôn dùng 2 singleton này, không tạo instance mới
from src.core.config import settings      # Settings
from src.core.database import db_instance # VectorDatabase (Qdrant + embedding model)
```

`db_instance` được khởi tạo 1 lần khi module load. Nó chứa:
- `db_instance.client` — QdrantClient
- `db_instance.embedding_model` — BAAI/bge-m3 SentenceTransformer

---

## Config Reference

Tất cả config đều qua `src/core/config.py`. Giá trị đọc từ `.env`:

```bash
# Bắt buộc
GROQ_API_KEY=gsk_xxx

# Optional
OPENAI_API_KEY=sk_xxx
QDRANT_SERVER_URL=http://qdrant:6333   # nếu dùng Qdrant server
REDIS_URL=redis://localhost:6379        # nếu dùng Redis cache

# RAG Tuning
TOP_K_RETRIEVAL=15
TOP_K_RERANK=3
SEMANTIC_CACHE_THRESHOLD=0.92
CHUNK_SIZE=800
CHUNK_OVERLAP=150

# Models
EMBEDDING_MODEL_NAME=BAAI/bge-m3
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_PROVIDER=groq                       # groq | openai | ollama
```

---

## Thêm tính năng mới

### Thêm một retriever mới
1. Tạo file `src/engine/retrieval/my_retriever.py`
2. Implement `search(query: str, collection_name: str) -> List[Dict[str, Any]]`
3. Return format chuẩn: `[{"id", "content", "metadata", "distance"}]`
4. Wire vào `HybridRetriever` trong `hybrid_search.py`

### Thêm một metric RAGAS mới
1. Import metric từ `ragas.metrics` trong `ragas_evaluator.py`
2. Thêm vào list `metrics=[...]` trong `_compute_ragas_metrics()`
3. Thêm tên metric vào `metric_cols` list

### Thêm một endpoint mới
1. Thêm vào `src/api/main.py`
2. Dùng `AIStore.*` thay vì khởi tạo engine mới
3. Thêm schema vào `src/schema/models.py` nếu cần

### Thêm một config mới
1. Thêm field vào `Settings` trong `src/core/config.py`
2. Đặt default value hợp lý
3. Dùng `SecretStr` cho sensitive values (API keys)

---

## Mocking Pattern (Tests)

Đây là pattern bắt buộc khi viết test — import module SAU khi declare fixture:

```python
@pytest.fixture(autouse=True)
def mock_heavy_dep(mocker):
    mock = MagicMock()
    mocker.patch("src.engine.retrieval.dense_search.db_instance", mock)
    return mock

# Import AFTER fixture
from src.engine.retrieval.dense_search import DenseRetriever

def test_something(mock_heavy_dep):
    mock_heavy_dep.embedding_model.encode.return_value = [MagicMock()]
    mock_heavy_dep.client.query_points.return_value.points = []
    ...
```

**Tại sao**: `db_instance` và models được load khi import module. Phải patch trước khi import để tránh load GPU model thật trong CI.

---

## Error Handling Conventions

| Scenario | Behavior |
|----------|----------|
| User input không hợp lệ | `raise ValueError("message")` |
| Retrieval thất bại | `return []` — graceful degradation |
| LLM call thất bại | Retry với exponential backoff (max 3 lần) |
| LLM hết retry | `raise RuntimeError("LLM call thất bại")` |
| Cache miss | `return None` — caller fallback sang pipeline |
| Qdrant connection error | Log error, return `[]` |

---

## Logging Convention

```python
from src.core.logger import logger

logger.info(f"[ClassName] Action: {variable}")      # Normal flow
logger.warning(f"[ClassName] Recoverable: {e}")    # Degraded but ok
logger.error(f"[ClassName] Fatal: {e}")            # Need attention
```

**Không dùng `print()`** trong production code.

---

## Common Gotchas

### 1. Qdrant file lock
Local Qdrant chỉ cho 1 process. Nếu API server đang chạy → không thể chạy script khác cùng lúc.
```bash
# Kiểm tra process đang lock
ps aux | grep uvicorn
kill <PID>
```

### 2. BM25 không persist
`SparseRetriever` build BM25 index fresh mỗi lần từ Qdrant scroll. Cache trong-memory reset khi restart. → Đây là lý do cần upgrade sang SPLADE (xem `FUTURE_UPGRADES.md`).

### 3. Stream meta frame format
```
# Frontend parse JSON meta ở cuối stream:
"\n\n__META__{\"sources\": [...], \"facts\": [...], \"session_id\": \"...\", \"cached\": false}"
```
Parse bằng cách split trên `"__META__"` → `JSON.parse()` phần sau.

### 4. Visual Frame RAG cần GEMINI_API_KEY
`chunk_type: "visual"` chunks chỉ được tạo khi `GEMINI_API_KEY` được set. Nếu không có key, frame extraction bị skip — pipeline vẫn hoạt động bình thường với text-only.

---

## Testing

```bash
# Chạy tất cả unit tests
python -m pytest tests/unit/ -v --tb=short

# Chạy với coverage
python -m pytest tests/unit/ --cov=src --cov-report=term-missing

# Chạy 1 file cụ thể
python -m pytest tests/unit/test_dense_search.py -v

# RAGAS benchmark (cần Qdrant data)
python tests/run_benchmark.py --generate --collection <name>
python tests/run_benchmark.py --evaluate --collection <name>
```

**Test count**: 208 unit tests, 0 integration tests  
**Coverage**: ~83% (CI enforces minimum 60%)

---

## Development Workflow

```bash
# 1. Setup
poetry install

# 2. Config
cp .env.example .env
# Điền GROQ_API_KEY=gsk_xxx

# 3. Chạy API
uvicorn src.api.main:app --reload --port 8000

# 4. Chạy Frontend
cd frontend && npm run dev

# 5. Ingest video test
python -c "
from src.engine.ingestion.pipeline import IngestionPipeline
p = IngestionPipeline(use_contextual_enrichment=False)
print(p.run('https://youtube.com/watch?v=VIDEO_ID'))
"

# 6. Trước khi commit
poetry run ruff check src/ tests/
poetry run pytest tests/unit/ -v
poetry run bandit -r src/ -ll
```
