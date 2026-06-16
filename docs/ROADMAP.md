# YouRAG — Roadmap

> **Tầm nhìn:** Nền tảng học tập và nghiên cứu thông minh dựa trên video YouTube — giúp người dùng hiểu sâu, ghi nhớ lâu và khai thác tri thức từ bất kỳ video nào chỉ bằng cách hỏi đáp tự nhiên.
>
> Updated 2026-06-16.

---

## Vision

```
Người dùng paste YouTube URL
    ↓
YouRAG index toàn bộ tri thức trong video (transcript + visual)
    ↓
Hỏi bất kỳ câu hỏi nào → nhận câu trả lời có nguồn, có timestamp
    ↓
Tự động tạo quiz, flashcard, tóm tắt để ôn tập
    ↓
So sánh, tổng hợp kiến thức từ nhiều video / cả khoá học
```

Đối tượng: **sinh viên, researcher, content creator, lifelong learner** — ai cũng học từ YouTube nhưng không có công cụ để học *hiệu quả*.

---

## Current State — What's Done

### Core RAG Pipeline
- [x] YouTube transcript extraction (pytubefix + youtube-transcript-api)
- [x] Whisper STT fallback khi video không có CC
- [x] Semantic chunking (pause-aware + vector valleys)
- [x] Late chunking embeddings (Jina jina-embeddings-v3, optional)
- [x] Contextual enrichment (LLM-generated context per chunk, optional)
- [x] Graph RAG (rule-based entity extraction, NetworkX)
- [x] Hybrid retrieval: Dense (Qdrant HNSW) + Sparse (BM25), chạy parallel
- [x] RRF fusion (alpha=0.5, rrf_k=60)
- [x] CrossEncoder reranker (mmarco-mMiniLMv2)
- [x] True LLM streaming
- [x] Semantic cache (Qdrant-backed)
- [x] Self-correction + citation grounding

### Học tập & Nghiên cứu — đã có
- [x] Hỏi đáp với timestamp citations — click → seek đúng vị trí trong video
- [x] YouTube IFrame embed tích hợp trong app
- [x] Multi-video chat — hỏi đồng thời nhiều video, so sánh nội dung
- [x] Suggested questions — gợi ý câu hỏi tiếp theo tự động
- [x] Video summary — tóm tắt toàn bộ nội dung video
- [x] Chat history — lưu lại lịch sử hỏi đáp theo session
- [x] Quiz & Flashcard generation — MCQ 10 câu + Anki-style flashcard, "Xem trong video" modal
- [x] Visual Frame RAG — extract frame mỗi ~30s, Gemini Flash mô tả visual, nhúng vào Qdrant (`chunk_type: "visual"`)

### Infrastructure
- [x] FastAPI backend + Next.js 14 frontend
- [x] Qdrant vector database (HNSW, cosine, 1024-dim)
- [x] Redis: job store, semantic cache, chat history, graph cache
- [x] PostgreSQL: chat sessions
- [x] Redis Streams ingest queue (crash recovery)
- [x] Docker Compose full stack (one-command deploy)
- [x] Rate limiting + API key auth

### Observability (đầy đủ 3 pillars)
- [x] Prometheus metrics + Grafana dashboard (auto-provisioned)
- [x] Loki log aggregation (Promtail)
- [x] Langfuse LLM tracing (self-hosted, port 3002)
- [x] MLflow experiment tracking (RAGAS benchmark, port 5001)
- [x] Grafana alerting (4 rules: error rate, latency, memory, uptime)

### LLM & Evaluation
- [x] Groq llama-3.3-70b-versatile (primary) + llama-3.1-8b-instant (fast)
- [x] OpenAI fallback
- [x] RAGAS benchmark (faithfulness, answer_relevancy, context_precision, context_recall)

---

## Roadmap — Học tập & Nghiên cứu

### Phase 1 — Công cụ học tập cốt lõi

#### 1.1 Playlist / Channel Bulk Ingest ⭐ *(chưa làm)*
**Impact: ★★★★★ | Effort: 1 ngày**

Paste URL của cả playlist hoặc kênh → YouRAG tự động ingest toàn bộ → instant research library cho cả khoá học hay podcast series. Kết hợp với multi-video chat để hỏi đáp xuyên suốt khoá học.

```
POST /ingest/playlist
{ "url": "youtube.com/playlist?list=PLxxx", "max_videos": 30 }
→ ingest song song, progress bar từng video
```

**Use case:** Sinh viên paste cả playlist bài giảng → hỏi "giải thích khái niệm X ở bài nào?" → YouRAG tìm đúng video và timestamp.

---

#### 1.2 Quiz & Flashcard Generation ✅ *DONE*
**Impact: ★★★★★ | Effort: 1 ngày**

Nút "Generate Quiz" và "Create Flashcards" trên mỗi video:
- **Quiz** (MCQ): 10 câu hỏi trắc nghiệm với 4 lựa chọn, đáp án có giải thích + timestamp nguồn
- **Flashcards** (Anki-style): mặt trước = khái niệm, mặt sau = giải thích + ví dụ từ video
- Export: JSON (Anki import), CSV, PDF

```
GET /quiz/{collection}?type=mcq&count=10
GET /flashcards/{collection}?count=20
```

**Use case:** Học xong video → generate quiz → ôn tập → biết mình hiểu đến đâu, chỗ nào cần xem lại.

---

#### 1.3 Visual Frame RAG ✅ *DONE*
**Impact: ★★★★★ | Effort: 3-4 ngày**

Text-only RAG bỏ sót 100% nội dung hình ảnh — slides, biểu đồ, code trên màn hình, công thức. Với video học thuật, 30-50% tri thức nằm trên visual.

Pipeline:
1. Extract frame mỗi ~20-30s bằng `yt-dlp` + `ffmpeg`
2. Vision model (Gemini Flash 1.5) mô tả frame: "slide này trình bày công thức backpropagation..."
3. Nhúng mô tả vào Qdrant cùng collection (`chunk_type: "visual"`, `frame_url`, `timestamp`)
4. Retrieval tự nhiên tìm được cả text lẫn visual chunks

```
"slide về attention mechanism" → tìm đúng frame → trả lời kèm thumbnail + timestamp
```

**Use case:** Sinh viên hỏi "công thức softmax ở đâu?" → YouRAG trả về đúng slide có công thức đó.

---

#### 1.4 Chapter-aware Chunking
**Impact: ★★★★ | Effort: 1 ngày**

YouTube auto-chapters (timestamp + title) là ranh giới ngữ nghĩa tự nhiên — dùng làm chunk boundary thay vì pause-based hiện tại. Lưu `chapter_title` vào metadata → hiển thị trong source chips → người dùng biết ngay câu trả lời đến từ phần nào của video.

```
Source: [03:42–07:15] "Chapter: Attention Is All You Need"
```

---

#### 1.5 Study Notes Export
**Impact: ★★★★ | Effort: 1 ngày**

Sau buổi học (nhiều Q&A về một video), tổng hợp toàn bộ hội thoại thành study notes có cấu trúc:
- Các khái niệm chính đã hỏi (với giải thích)
- Timeline: khái niệm nào ở phút nào
- Gaps: "bạn chưa hỏi về..." (dựa trên chapter list)

Export ra Markdown (Obsidian-ready), Notion, hoặc PDF.

```
GET /study-notes/{session_id}?format=markdown
```

---

### Phase 2 — Nghiên cứu nâng cao

#### 2.1 Comparative Analysis Mode
**Impact: ★★★★★ | Effort: 2 ngày**

Mode so sánh chuyên biệt: chọn 2-5 video về cùng chủ đề → AI tự động tổng hợp bảng so sánh quan điểm, cách giải thích, điểm đồng/khác nhau. Không chỉ merge search kết quả mà có prompt template đặc biệt cho comparison.

```
"So sánh cách 3 video này giải thích gradient descent"
→ Bảng: Video A vs B vs C | Cùng quan điểm | Khác nhau | Ai đúng hơn?
```

**Use case:** Researcher so sánh nhiều nguồn trước khi viết paper. Sinh viên chọn video nào để học.

---

#### 2.2 Speaker Diarization
**Impact: ★★★★ | Effort: 2-3 ngày**

Dùng `pyannote-audio` để tách speaker → tag mỗi chunk bằng `speaker_id`. Có thể query theo speaker cụ thể.

```
"Chỉ lấy câu trả lời của khách mời về AI safety"
"Host nói gì về vấn đề này?"
```

**Use case:** Podcast, interview, panel discussion — tìm đúng quan điểm của đúng người.

---

#### 2.3 Clip Export
**Impact: ★★★★ | Effort: 2 ngày**

Click source chip → download đoạn clip đó ra MP4 (`yt-dlp --download-sections`). Biến YouRAG thành research + content tool.

```
GET /clip/{collection}?start=180&end=240
→ stream MP4 30 giây đó về browser
```

**Use case:** Content creator tìm clip để trích dẫn. Giáo viên cắt đoạn video cho bài giảng.

---

#### 2.4 Learning Progress Tracking
**Impact: ★★★★ | Effort: 2 ngày**

Theo dõi tiến trình học:
- Video nào đã xem, đã hỏi bao nhiêu câu
- Khái niệm nào đã hiểu (dựa trên Q&A history)
- Gợi ý "bạn chưa tìm hiểu về X" (từ chapters chưa được hỏi)
- Streak, thời gian học theo ngày (dashboard)

---

### Phase 3 — Platform & Collaboration

#### 3.1 Multi-user Auth (JWT)
Mỗi user có account riêng, collection riêng, history riêng. Chia sẻ collection với người khác.

#### 3.2 Shared Study Rooms
Nhiều người cùng học một video — share câu hỏi, highlight, quiz với nhau. Collaborative annotations trên timeline video.

#### 3.3 Async Ingest Progress (SSE)
Thay polling 2s bằng Server-Sent Events → push real-time progress (transcript X%, chunking X%, embedding X%).

#### 3.4 Mobile-first UI
Responsive redesign cho điện thoại — hiện tại đã có tab Chat/Video nhưng UX mobile còn thô.

---

### Phase 4 — Intelligence & Personalization

#### 4.1 Adaptive Question Generation
Dựa trên lịch sử Q&A của user → tự động generate câu hỏi ở độ khó phù hợp (spaced repetition). Biết user đã hiểu gì → tập trung vào gaps.

#### 4.2 Model Routing
Simple queries → llama-3.1-8b-instant (5x faster)
Complex synthesis/comparison → llama-3.3-70b-versatile
Heuristic: query length + keywords ("so sánh", "phân tích", "tổng hợp")

#### 4.3 RAG Feedback Loop
👍/👎 trên mỗi câu trả lời → upweight chunks được feedback tốt → retrieval tự cải thiện theo thời gian mà không cần re-train.

#### 4.4 A/B Testing RAG Configs
50% traffic → Hybrid tier vs 50% → Advanced tier → đo user satisfaction thực tế, không chỉ RAGAS benchmark.

#### 4.5 Online LLM-as-Judge
Auto-score câu trả lời production (nhẹ hơn RAGAS, real-time) → alert khi quality drop → log vào MLflow.

---

### Phase 5 — Scale & Deploy

#### 5.1 CD Pipeline (Railway / Fly.io)
```
git tag v1.x.x && git push --tags → auto deploy
```

#### 5.2 K3s / Helm Chart
Single-node Kubernetes, HPA scale backend 1→3 pods khi CPU > 70%.

#### 5.3 Embedding Drift Detection
Alert khi video content domain thay đổi → cosine similarity distribution shift → trigger re-embed.

---

## Feature Priority Matrix

| Feature | Impact học tập | Effort | Phase |
|---------|---------------|--------|-------|
| Playlist bulk ingest | ★★★★★ | 1 ngày | 1 |
| Quiz / Flashcard | ★★★★★ | ✅ Done | 1 |
| Visual Frame RAG | ★★★★★ | ✅ Done | 1 |
| Comparative analysis | ★★★★★ | 2 ngày | 2 |
| Chapter-aware chunking | ★★★★ | 1 ngày | 1 |
| Study Notes Export | ★★★★ | 1 ngày | 1 |
| Learning Progress | ★★★★ | 2 ngày | 2 |
| Speaker Diarization | ★★★★ | 2-3 ngày | 2 |
| Clip Export | ★★★ | 2 ngày | 2 |
| Multi-user Auth | ★★★ | 3 ngày | 3 |
| Shared Study Rooms | ★★★ | 4 ngày | 3 |
| Adaptive Questions | ★★★★★ | 3 ngày | 4 |
| Model Routing | ★★★ | 1 ngày | 4 |
| RAG Feedback Loop | ★★★ | 2 ngày | 4 |

---

## Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| Groq 100K tokens/day free tier | Benchmark cần 2 accounts | Key rotation đã implement |
| BGE-reranker ~568MB RAM | Slow cold start | Model cached sau lần đầu |
| Whisper STT chỉ chạy CPU | Video > 30min rất chậm | Chỉ dùng khi không có CC |
| Text-only RAG (legacy) | Đã có Visual Frame RAG | `GEMINI_API_KEY` cần set |
| 1 API key cho toàn bộ | Không có user isolation | Multi-user Auth — Phase 3 |
