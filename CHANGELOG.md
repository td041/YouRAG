# Changelog

All notable changes to YouRAG are documented here.

---

## [Unreleased] — 2026-06-16

### Added
- **Visual Frame RAG** — extract video frames every ~30s, Gemini Flash 1.5 describes visual content (slides, diagrams, code), embeds into Qdrant as `chunk_type: "visual"`
- **Multi-video chat** — `/chat/stream` accepts `collections: [...]` list; `search_multi()` fans out to N collections and RRF-fuses results
- **Quiz & Flashcard generation** — `GET /quiz/{collection}?mode=quiz|flashcard&count=N` with retry logic and anti-hallucination prompts; `_inject_start_times()` parses timestamp strings server-side
- **`/learn` page** — 3D CSS flip flashcards, MCQ quiz with score, "Xem trong video" YouTube IFrame seek modal
- **Library, Analytics, Settings pages** — App Router multi-page layout
- **Whisper Vietnamese language detection** — counts diacritical chars in title/description, passes `language="vi"` hint to faster-whisper to prevent misidentification as Indonesian/Thai
- **Grafana alerting** — 4 rules auto-provisioned: error rate >5%, p95 latency >3s, memory >90%, uptime
- BGE-reranker-v2-m3 replacing mmarco (better multilingual, better Vietnamese)
- RAGAS benchmark with Mistral evaluator, multilingual embeddings, round-robin Groq key rotation

### Changed
- Reranker now falls back to CPU instead of disabling (208 unit tests, all pass)
- CORS restricted to `ALLOWED_ORIGINS` env var (default: localhost:3000,3001); removed `allow_credentials=True`
- Graph store unified to JSON throughout — delete endpoint was last holdout using `.gpickle`
- Reranker top_k: 5 → 9 (better context recall)
- GRAPH_MAX_CHUNKS: 15 → 40 (better entity coverage for long videos)

### Fixed
- `graph_data.get()` AttributeError when `graph_retriever` is None (returned `[]` instead of `{}`)
- `AIStore.cache` None crash in both stream and non-stream chat endpoints
- `format_timestamp(c['metadata']['start_time'])` KeyError on malformed chunks (safe `.get()`)
- Graph retriever `_graph_cache.pop()` crash when `graph_retriever` is None
- mm:ss timestamp citations in quiz/flashcard always showing "0:01" (LLM was copying example value)

### Removed
- Gemini production fallback (Groq rotation handles rate limits)

---

## [1.2.0] — 2026-05-31

### Added
- **Graph store + contextual cache migrated to Redis** (no more data loss on restart)
- `src/core/redis_client.py` singleton shared across modules
- Proactive BM25 cache invalidation on re-ingest

### Fixed
- Cross-collection semantic cache contamination (scoped by `collection_name`)
- Redis global init causing startup failure when Redis is down
- Auth bypass with whitespace-only `API_KEY`
- Summarizer `IndexError` on empty collection
- YouTube loader hardcoded 60s timeout → configurable `YOUTUBE_FETCH_TIMEOUT`
- Streaming generator silently swallowing exceptions
- Citation regex only matching `[M:SS]` — now supports `[H:MM:SS]`
- Graph RAG `NoneType.strip()` crash on null triples from LLM

---

## [1.1.0] — 2026-05-24

### Added
- **Streaming JSON meta frame** replacing fragile `__SOURCES__::` string delimiters
- **Redis job store** for ingest jobs (TTL 24h, fallback in-memory)
- **`/health` endpoint** — checks Qdrant, Redis, Generator, Reranker
- **Rate limiting** via `slowapi` (20 req/min chat, 5 req/min ingest)
- **YouTube URL validation** in backend (HTTP 422 on invalid)
- **`MISTRAL_EVAL_API_KEY`** for RAGAS benchmark (separate from production quota)
- Docker build migrated from Poetry → **uv** (build time 10min → 3min)

### Changed
- ZenML dead code removed from `pipeline.py` (~200ms startup improvement)
- Contextual cache disk pruning (30-day TTL for local fallback files)

---

## [1.0.0] — 2026-05-21

### Added
- **API Key authentication** (`X-API-Key` header) for all write endpoints
- **Gemini auto-fallback** when Groq hits rate limit (429)
- **Full UX overhaul**: dark/light mode, mobile responsive, dynamic suggestions
- **Semantic cache scoped by collection** (prevents cross-video contamination)
- **Citation Grounding**: removes fabricated `[mm:ss]` timestamps from answers
- **Stream self-correction**: generate draft → audit vs graph facts → stream corrected answer
- **Whisper STT fallback** for videos without captions
- **Jina Late Chunking** support (`use_late_chunking=true`)
- Persistent chat history (Redis + PostgreSQL dual-layer)
- Docker Compose with 5 services (backend, frontend, Qdrant, PostgreSQL, Redis)
- GitHub Actions CI: Ruff + Bandit + Pytest (83% coverage)

### Architecture
- HybridRetriever (Dense + BM25 + RRF fusion)
- CrossEncoder Reranker (mmarco-mMiniLMv2)
- Knowledge Graph (NetworkX, JSON serialization)
- Self-Correction RAG with graph-fact auditing
