# YouRAG — System Architecture

> Version: 1.3.0 | Last updated: 2026-06-16

---

## Overview

YouRAG is a production-grade **Retrieval-Augmented Generation (RAG)** system that transforms any YouTube video into a queryable knowledge base with timestamp-cited answers.

The pipeline is **ablation-tested** — each layer (Naive → Hybrid → Advanced) is independently measured with RAGAS before inclusion.

---

## Pipeline

```
[YouTube URL]
    ↓
YouTubeLoader (pytubefix + youtube-transcript-api)
    ↓ [auto-fallback if no captions]
WhisperTranscriber (faster-whisper, local STT)
    ↓
SemanticChunker
  Phase 1: Pause-aware atomic splitting (>1.5s pause = topic break)
  Phase 2: Vector semantic valley detection (dynamic percentile threshold)
    ↓ [optional]
ContextualEnricher (LLM prefix injection per chunk — Anthropic technique)
    ↓ [default ON]
LateChunkingEmbedder (Jina jina-embeddings-v3, context-aware 1024-dim)
    ↓
GraphExtractor (LLM triple extraction → NetworkX DiGraph)
    ↓
Qdrant VectorDB (BAAI/bge-m3, 1024-dim, cosine HNSW)
    ↓ Redis (graph store, contextual cache, job store, semantic cache)
    ↓
━━━━━━━━━━━━━━━━━━ QUERY TIME ━━━━━━━━━━━━━━━━━━
    ↓
HybridRetriever
  Dense: Qdrant vector search (bge-m3)
  Sparse: BM25 in-memory index
  Fusion: RRF (alpha=0.5, k=60)
    ↓
CrossEncoderReranker (BAAI/bge-reranker-v2-m3, top_k=9)
    ↓
AnswerGenerator
  1. Build prompt (context + graph facts + summary)
  2. LLM call → draft answer (llama-3.3-70b via Groq)
  3. Self-correction (audit draft vs graph facts)
  4. Citation grounding (remove fabricated [mm:ss])
    ↓
Final Answer + [mm:ss] Citations + Sources
```

---

## Storage Architecture

| Data | Storage | TTL |
|---|---|---|
| Vector embeddings | Qdrant (Docker/Cloud) | Permanent |
| Knowledge graphs | Redis `graph:{collection}` | 90 days |
| Contextual cache | Redis `ctx_cache:{hash}` | 30 days |
| Semantic cache | Qdrant collection `semantic_cache` | Permanent |
| Ingest job state | Redis `ingest_job:{id}` | 24 hours |
| Chat history | PostgreSQL + Redis | Permanent / 24h cache |

---

## Anti-Hallucination Stack

1. **Strict system prompt** — "If not in documents, say you don't know"
2. **Self-correction** — Second LLM call audits draft against Knowledge Graph facts
3. **Citation grounding** — Validates each `[mm:ss]` against retrieved chunk timestamps
4. **No-context early return** — Returns "not found" immediately when no chunks retrieved
5. **Stream self-correction** — When graph facts present: draft → correct → then stream

---

## Ablation Study Results

| Metric | Naive (Dense) | Hybrid (RRF) | Advanced (Rerank) |
|---|---|---|---|
| Faithfulness | 0.851 | **0.929** | 0.843 |
| Answer Relevancy | 0.767 | **0.832** | 0.782 |
| Context Precision | **0.936** | 0.935 | 0.902 |
| Context Recall | **1.000** | 0.988 | 0.963 |
| Factual Correctness | 0.733 | **0.777** | 0.737 |

*Evaluated on 20 questions using Mistral Small evaluator + multilingual embeddings.*

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI 0.110 + Uvicorn |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Vector DB | Qdrant (HNSW cosine, 1024-dim) |
| Embeddings | BAAI/bge-m3 (production) + Jina v3 (late chunking) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | Groq llama-3.3-70b-versatile |
| Knowledge Graph | NetworkX (JSON, no pickle) |
| Cache | Redis (graph, contextual, semantic, jobs) |
| Chat History | PostgreSQL + Redis |
| Build | uv (Docker), Poetry (local/CI) |
| CI/CD | GitHub Actions |

---

## Singletons

```python
from src.core.database import db_instance   # VectorDatabase (Qdrant + bge-m3)
from src.core.config import settings        # Pydantic Settings
from src.core.redis_client import get_redis # Lazy Redis client (returns None if down)
```

## Frontend Pages (Next.js App Router)

| Route | Purpose |
|---|---|
| `/` | Chat + Video panel (main RAG interface) |
| `/library` | Manage ingested videos, delete, rebuild graph |
| `/learn` | Quiz (MCQ) + Flashcard (3D flip) per video |
| `/analytics` | Prometheus metrics via Grafana embed |
| `/settings` | System config overview |

## Key Design Decisions

- **JSON over pickle** for graph store — eliminates RCE risk
- **Redis-first storage** with local file fallback — survives Redis downtime
- **BM25 in-memory** with 2000-doc cap — prevents OOM on large collections
- **Round-robin Groq keys** for benchmark — shares daily quota evenly across accounts
- **Visual Frame RAG** skips gracefully when `GEMINI_API_KEY` not set — text-only fallback
- **CORS restricted** to `ALLOWED_ORIGINS` env var — no wildcard + credentials in production
- **Reranker on CPU** — warns but loads; GPU preferred, CPU tolerated for dev/CI
