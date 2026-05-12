<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white" alt="Next.js"/>
  <img src="https://img.shields.io/badge/LLM-Llama_3.3_70B-7C3AED?logo=meta&logoColor=white" alt="LLM"/>
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=qdrant&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white" alt="CI"/>
</p>

# 🧠 YouRAG — Advanced YouTube RAG with Knowledge Graph & Self-Correction

**YouRAG** là hệ thống **Retrieval-Augmented Generation (RAG)** thế hệ mới, biến bất kỳ video YouTube nào thành một nguồn tri thức có thể truy vấn và hỏi đáp bằng AI ở mức chuyên gia. Hệ thống kết hợp **Knowledge Graph**, **Hybrid Search**, **Cross-Encoder Reranking** và **Graph-based Self-Correction** — những kỹ thuật SOTA nhất trong lĩnh vực RAG hiện nay.

> **Không chỉ là chatbot video.** YouRAG xây dựng một "bộ não kỹ thuật số" cho mỗi video: trích xuất thực thể, dựng đồ thị quan hệ, và bắt buộc AI phải đối chiếu sự thật trước khi trả lời — **loại bỏ ảo giác (hallucination) ở mức kiến trúc.**

---

## ⚡ Kiến trúc Hệ thống (System Architecture)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          🎬 YOUTUBE VIDEO URL                              │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   YouTubeLoader          │  pytubefix + youtube-transcript-api
                    │   (Parallel Fetch)       │  Tải metadata + transcript song song
                    └────────────┬─────────────┘
                                 │
            ┌────────────────────▼────────────────────┐
            │       SemanticChunker (SOTA 2-Phase)     │
            │                                          │
            │  Phase 1: Pause-Aware Atomic Splitting   │  Gộp dòng transcript → câu hoàn chỉnh
            │           (Pause > 1.5s = ngắt topic)    │  dựa trên khoảng lặng người nói
            │                                          │
            │  Phase 2: Vector Semantic Grouping       │  Nhúng vector (all-MiniLM-L6-v2)
            │           Dynamic Percentile Threshold   │  Dò "thung lũng ngữ nghĩa" để cắt chunk
            └────────────┬─────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼                               ▼
┌──────────────────┐          ┌─────────────────────────┐
│ ContextualEnricher│          │   GraphExtractor         │
│ (Anthropic Tech)  │          │   (Rule-based NER)       │
│                   │          │                           │
│ LLM bơm 1-2 câu  │          │ Trích xuất Entity +       │
│ ngữ cảnh gốc vào │          │ Relation → NetworkX Graph │
│ đầu mỗi chunk    │          │                           │
└────────┬──────────┘          └──────────┬────────────────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
         ┌─────────────────────────┐         ┌──────────────────┐
         │   Qdrant Vector DB       │         │  NetworkX Graph   │
         │   BAAI/bge-m3 (1024-dim) │         │  Knowledge Graph  │
         │   Cosine HNSW Index      │         │  (.gpickle)       │
         └─────────────────────────┘         └──────────────────┘
                      │                                │
═══════════════════════════════════════════════════════════════════
                      │         🔍 QUERY TIME          │
                      ▼                                ▼
         ┌─────────────────────┐          ┌──────────────────────┐
         │   HybridRetriever    │          │   GraphRetriever      │
         │                      │          │                       │
         │  Dense (Qdrant)      │          │  Multi-hop Traversal  │
         │  + Sparse (BM25)     │          │  Entity Extraction    │
         │  → RRF Fusion        │          │  Subgraph Matching    │
         │    (α=0.5, k=60)     │          │                       │
         └──────────┬───────────┘          └───────────┬───────────┘
                    │                                  │
                    ▼                                  │
         ┌─────────────────────┐                       │
         │ CrossEncoderReranker │                       │
         │ mmarco-mMiniLMv2     │                       │
         │ (Logit Scoring)      │                       │
         └──────────┬───────────┘                       │
                    │                                   │
                    └──────────────┬────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │     AnswerGenerator           │
                    │                               │
                    │  1. PromptBuilder → LLM Call   │  Llama-3.3-70B (Groq)
                    │  2. Draft Answer              │
                    │  3. 🔄 Self-Correction:       │  So sánh draft vs Graph Facts
                    │     AI audit + auto-fix        │  Phát hiện mâu thuẫn → sửa lỗi
                    │  4. Final Answer + [mm:ss]     │  Trích dẫn mốc thời gian
                    └──────────────────────────────┘
```

---

## 🔥 Điểm nhấn Kỹ thuật (Technical Highlights)

### 🧬 Semantic Chunking — Không còn cắt văn bản "bừa bãi"
Thay vì cắt theo số ký tự cố định (naive chunking), YouRAG sử dụng thuật toán **2 giai đoạn**:
1. **Pause-Aware Splitting**: Phân tích khoảng lặng giữa các dòng transcript (>1.5s = chuyển topic). Kết hợp LLM để phục hồi dấu câu cho transcript thô.
2. **Vector Semantic Valleys**: Nhúng vector từng câu, tính cosine similarity giữa các câu liền kề, sử dụng **Dynamic Percentile Threshold** (top 15% điểm rớt mạnh nhất) để xác định điểm cắt. Ngưỡng này **tự co giãn theo từng video**, không hardcode.

### 🕸️ Knowledge Graph RAG — Vượt xa Vector Search
Vector search chỉ tìm được "đoạn văn giống nhau". **Graph RAG** tìm được **quan hệ ẩn** mà vector bị mù:
- Trích xuất bộ ba `(Entity → Relation → Entity)` từ mỗi chunk
- Xây dựng đồ thị tri thức bằng **NetworkX**
- Hỗ trợ **multi-hop reasoning**: "A liên quan B, B liên quan C → A liên quan C"
- Cung cấp **Graph Facts** làm bằng chứng cho bước Self-Correction

### 🔍 Hybrid Search + Reciprocal Rank Fusion (RRF)
Kết hợp 2 paradigm tìm kiếm:
| Engine | Vai trò | Ưu điểm |
|--------|---------|---------|
| **Dense** (Qdrant + bge-m3) | Tìm theo ý nghĩa ngữ nghĩa | Hiểu đồng nghĩa, paraphrase |
| **Sparse** (BM25) | Tìm theo từ khóa chính xác | Bắt tên riêng, số liệu, thuật ngữ |

Hai nguồn kết quả được hòa trộn bằng thuật toán **RRF** (`α=0.5, k=60`) — công bằng, không thiên vị engine nào.

### ⚖️ Cross-Encoder Reranking — Bộ lọc chất lượng cuối cùng
Sau Hybrid Search, các ứng viên được chấm điểm lại bằng **mmarco-mMiniLMv2** (Cross-Encoder đa ngôn ngữ). Khác với Bi-Encoder (so khoảng cách vector), Cross-Encoder cho Query+Document vào model **cùng lúc** → điểm đánh giá chính xác tuyệt đối, loại bỏ nhiễu triệt để.

### 🧠 Graph-based Self-Correction — AI tự kiểm tra bản thân
Đây là tính năng **SOTA đỉnh cao** của YouRAG:
1. LLM tạo **bản nháp** câu trả lời
2. Hệ thống so sánh bản nháp với **Graph Facts** (sự thật từ đồ thị tri thức)
3. Nếu phát hiện **mâu thuẫn hoặc ảo giác** → LLM được gọi lần 2 để **tự sửa lỗi**
4. Chỉ trả về câu trả lời đã qua kiểm toán

> *"AI không chỉ trả lời — AI còn phải chứng minh câu trả lời không sai."*

### 💬 Persistent Chat History — Kiến trúc 2 tầng
Lịch sử trò chuyện được quản lý bằng kiến trúc **dual-layer**:
| Tầng | Công nghệ | Vai trò |
|------|-----------|---------|
| **Speed Layer** | Redis | Cache tốc độ cao, TTL 24h |
| **Durability Layer** | PostgreSQL | Lưu trữ vĩnh viễn, fallback khi Redis gặp sự cố |

Hệ thống tự động **backfill** cache Redis từ PostgreSQL khi xảy ra cache miss.

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Backend API** | FastAPI 0.110 + Uvicorn | RESTful API + Streaming Response |
| **Frontend** | Next.js 14 (React + TypeScript + Tailwind) | Glassmorphism UI, Dark Mode |
| **Vector DB** | Qdrant (Rust-based, HNSW) | Tìm kiếm vector siêu tốc |
| **Graph DB** | NetworkX | Đồ thị tri thức in-memory |
| **Relational DB** | PostgreSQL 15 | Chat history persistence |
| **Cache** | Redis 7 | Session cache, semantic cache |
| **Embedding** | `BAAI/bge-m3` (1024-dim, multilingual) | Nhúng vector đa ngôn ngữ |
| **Chunking Embed** | `all-MiniLM-L6-v2` | Semantic valley detection |
| **Reranker** | `mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder chấm điểm chéo |
| **LLM (Main)** | `llama-3.3-70b-versatile` via Groq | Generation + Self-Correction |
| **LLM (Fast)** | `llama-3.1-8b-instant` via Groq | Contextual Enrichment |
| **LLM (Fallback)** | OpenAI / Ollama | Dự phòng khi Groq không khả dụng |
| **MLOps** | ZenML | Pipeline orchestration + checkpointing |
| **Evaluation** | RAGAS Framework | LLM-as-a-judge benchmark |
| **CI/CD** | GitHub Actions | Lint (Ruff) + Security (Bandit) + Tests |
| **Containerization** | Docker Compose + BuildKit | Multi-service orchestration |
| **Dependency Mgmt** | Poetry | Deterministic builds via lockfile |

---

## 🏗️ Cấu trúc Dự án

```
YouRAG/
├── src/
│   ├── api/
│   │   └── main.py                    # FastAPI endpoints + AIStore startup
│   ├── core/
│   │   ├── config.py                  # Pydantic Settings (SecretStr, Singleton)
│   │   ├── database.py                # VectorDatabase singleton (Qdrant + bge-m3)
│   │   ├── postgres.py                # PostgreSQL engine + retry init
│   │   ├── logger.py                  # Centralized logging
│   │   └── utils.py                   # format_timestamp() utility
│   ├── engine/
│   │   ├── ingestion/
│   │   │   ├── pipeline.py            # ZenML pipeline + IngestionPipeline wrapper
│   │   │   ├── youtube_loader.py      # Parallel fetch (pytubefix + transcript API)
│   │   │   ├── chunker.py             # SemanticChunker (2-phase SOTA)
│   │   │   ├── contextual_enricher.py # Anthropic contextual retrieval technique
│   │   │   └── graph_extractor.py     # Rule-based entity/relation extraction
│   │   ├── retrieval/
│   │   │   ├── hybrid_search.py       # HybridRetriever (RRF fusion)
│   │   │   ├── dense_search.py        # DenseRetriever (Qdrant vector search)
│   │   │   ├── sparse_search.py       # SparseRetriever (BM25 in-memory)
│   │   │   └── graph_rag.py           # KnowledgeGraphBuilder + GraphRetriever
│   │   ├── generation/
│   │   │   ├── answer_generator.py    # AnswerGenerator (Self-Correction RAG)
│   │   │   ├── llm_client.py          # Multi-provider LLM (Groq/OpenAI/Ollama)
│   │   │   ├── prompt_builder.py      # Dynamic prompt (Standard/Mindmap/Table)
│   │   │   └── summarizer.py          # VideoSummarizer (full video summary)
│   │   ├── ranking/
│   │   │   └── cross_encoder.py       # CrossEncoderReranker (mmarco-mMiniLMv2)
│   │   └── chat/
│   │       └── history.py             # Dual-layer chat history (Redis + Postgres)
│   ├── models/
│   │   └── chat.py                    # SQLModel schemas (ChatSession, ChatMessage)
│   ├── schema/
│   │   └── models.py                  # Pydantic API models
│   └── cache/
│       └── semantic_cache.py          # SemanticCache (vector-based dedup)
├── frontend/                          # Next.js 14 UI
│   ├── app/                           # App Router (page.tsx, layout.tsx)
│   ├── components/                    # ChatPanel, VideoPanel, Sidebar
│   └── lib/                           # API client, types
├── tests/
│   ├── unit/                          # 14 test files, all mocked, CI-safe
│   ├── integration/                   # Real AI + DB tests (requires API keys)
│   └── benchmark/                     # RAGAS ablation study
├── .github/workflows/
│   ├── ci.yml                         # Lint + Security + Unit Tests + Integration
│   └── benchmark.yml                  # RAGAS evaluation pipeline
├── docker-compose.yml                 # 5 services: backend, frontend, qdrant, postgres, redis
├── Dockerfile                         # Poetry-based, layer-cached, GPU-ready
├── Makefile                           # 15+ shortcuts (build, up, rebuild, logs, test...)
├── pyproject.toml                     # Poetry dependencies + Ruff + Pytest config
└── poetry.lock                        # Deterministic dependency resolution
```

---

## 🚀 Khởi chạy Nhanh

### Yêu cầu
- Docker + Docker Compose
- NVIDIA GPU (khuyến nghị) hoặc CPU
- Groq API Key ([Lấy miễn phí tại đây](https://console.groq.com))

### 1. Clone & Cấu hình

```bash
git clone https://github.com/td041/YouRAG.git
cd YouRAG
make setup   # Tạo file .env từ template
```

Mở file `.env` và điền API key:
```env
GROQ_API_KEY=gsk_xxxxx
```

### 2. Khởi chạy Hệ thống

```bash
make rebuild   # Build + Start tất cả 5 services
```

### 3. Truy cập

| Service | URL | Mô tả |
|---------|-----|--------|
| **Frontend** | http://localhost:3000 | Giao diện chat |
| **Backend API** | http://localhost:8000 | REST API |
| **API Docs** | http://localhost:8000/docs | Swagger UI |

---

## 📋 Makefile Commands

```bash
make help           # Xem tất cả lệnh
make setup          # Tạo .env từ template
make build          # Build Docker images
make up             # Khởi chạy (background)
make down           # Dừng hệ thống
make rebuild        # Down → Build (cached) → Up
make rebuild-fresh  # Down → Build (no-cache) → Up
make logs           # Xem logs tất cả services
make logs-backend   # Xem logs backend
make status         # Trạng thái containers + GPU
make shell-backend  # SSH vào container backend
make shell-db       # Vào PostgreSQL CLI
make test           # Chạy unit tests
make lint           # Kiểm tra code (Ruff)
make clean          # Xóa containers
make clean-all      # Xóa tất cả kể cả data ⚠️
```

---

## 🧪 Testing & Quality

### Unit Tests (14 files, ~60% coverage)
```bash
poetry run pytest tests/unit/ -v --cov=src --cov-report=term-missing
```
- Tất cả external I/O được mock (Qdrant, Redis, Postgres, LLM APIs)
- Chạy trong < 2 giây, CI-safe

### Integration Tests
```bash
poetry run pytest tests/integration/ -v
```
- Yêu cầu Groq API key và Docker services đang chạy

### Code Quality
```bash
poetry run ruff check src/ tests/    # Lint
poetry run bandit -r src/ -ll        # Security scan
poetry run mypy src/ --ignore-missing-imports  # Type check
```

### RAGAS Benchmark (Ablation Study)
```bash
poetry run python tests/run_benchmark.py --evaluate
```
Đo lường 4 chỉ số vàng qua từng tầng SOTA:

| Metric | Naive RAG | + Hybrid | + Reranker | + Self-Correction |
|--------|-----------|----------|------------|-------------------|
| **Faithfulness** | Baseline | ↑ | ↑↑ | ↑↑↑ |
| **Answer Relevancy** | Baseline | ↑ | ↑↑ | ↑↑↑ |
| **Context Precision** | Baseline | ↑↑ | ↑↑↑ | ↑↑↑ |
| **Context Recall** | Baseline | ↑ | ↑↑ | ↑↑ |

---

## 🐳 Docker Architecture

```yaml
# 5 services, 1 network, GPU-ready
services:
  qdrant       # Vector DB (port 6333)
  postgres     # Chat history (port 5432)
  redis        # Session cache (port 6379)
  backend      # FastAPI + AI Models (port 8000, GPU)
  frontend     # Next.js UI (port 3000)
```

### Tối ưu hóa Docker
- **Layer Caching**: Dependencies được cài trước source code → rebuild chỉ mất ~5s khi chỉ sửa code
- **Model Caching**: Mount `~/.cache/huggingface` từ host → không tải lại ~4.8GB models mỗi lần start
- **Poetry-based**: Sử dụng `poetry.lock` để đảm bảo reproducible builds
- **BuildKit**: Bật `DOCKER_BUILDKIT=1` để build song song

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `GET` | `/` | Health check |
| `GET` | `/collections` | Danh sách video đã ingest |
| `POST` | `/ingest` | Ingest video YouTube mới |
| `POST` | `/chat` | Chat RAG (sync) |
| `POST` | `/chat/stream` | Chat RAG (streaming) |
| `GET` | `/history/{session_id}` | Lấy lịch sử chat |
| `POST` | `/graph/build/{collection}` | Build/Rebuild Knowledge Graph |
| `GET` | `/summarize/{collection}` | Tóm tắt video |
| `GET` | `/summarize/stream/{collection}` | Tóm tắt video (streaming) |

---

## 🗺️ Roadmap

- [x] Semantic Chunking (Pause-aware + Vector valleys)
- [x] Contextual Enrichment (Anthropic technique)
- [x] Knowledge Graph (NetworkX + Rule-based NER)
- [x] Hybrid Search (Dense + BM25 + RRF)
- [x] Cross-Encoder Reranking (mmarco-mMiniLMv2)
- [x] Graph-based Self-Correction
- [x] Streaming Response + [mm:ss] Citations
- [x] Persistent Chat History (Redis + PostgreSQL)
- [x] Next.js Frontend (Glassmorphism, Dark Mode)
- [x] Docker Compose (5 services, GPU-ready)
- [x] CI/CD (GitHub Actions: Lint + Security + Tests)
- [x] RAGAS Ablation Benchmark
- [ ] Semantic Caching (GPTCache/Redis Vector)
- [ ] Semantic Router (Intent classification)
- [ ] Observability (Phoenix/LangSmith tracing)
- [ ] Multi-video cross-referencing

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

<p align="center">
  <b>Built with 🔥 by <a href="https://github.com/td041">td041</a></b>
</p>
