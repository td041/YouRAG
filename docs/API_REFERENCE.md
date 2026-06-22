# YouRAG — API Reference

> Base URL: `http://localhost:8000` (dev) | `https://your-backend.up.railway.app` (production)  
> Version: 1.3.0

---

## Authentication

Protected endpoints require the `X-API-Key` header.

```http
X-API-Key: your_api_key_here
```

**Dev mode:** If `API_KEY` is not set in `.env`, all endpoints are open (no auth required).

---

## Endpoints

### `GET /`

Status check.

```json
{"status": "online", "message": "YouRAG API is ready."}
```

---

### `GET /health`

Deep health check — used by Railway/Docker healthchecks.

**Response 200 (healthy):**
```json
{
  "status": "healthy",
  "checks": {
    "qdrant": "ok",
    "redis": "ok",
    "generator": "ok",
    "reranker": "ok"
  }
}
```

**Response 503 (degraded):** Same structure with error details per service.

---

### `GET /collections`

List all ingested videos.

```json
[
  {
    "name": "ly-thuyet-tro-choi",
    "title": "Lý Thuyết Trò Chơi | Kraven",
    "video_id": "foO2lKeumhk"
  }
]
```

---

### `POST /ingest`

🔒 **Protected**

Start async ingestion of a YouTube video.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "use_contextual": false,
  "use_late_chunking": true
}
```

**Response:**
```json
{"job_id": "uuid-v4", "status": "queued"}
```

**Rate limit:** 5 requests/minute per IP.

---

### `GET /ingest/status/{job_id}`

Poll ingest job status.

**States:** `queued` → `running` → `done` / `error`

**Response (done):**
```json
{
  "status": "done",
  "result": {
    "video_id": "foO2lKeumhk",
    "title": "Lý Thuyết Trò Chơi",
    "collection_name": "ly-thuyet-tro-choi",
    "chunks_added": 17,
    "late_chunking_used": true,
    "latency": {"extract_s": 2.1, "chunk_s": 8.3, "total_s": 14.6}
  }
}
```

---

### `POST /chat`

🔒 **Protected**

RAG chat (synchronous).

**Request:**
```json
{
  "query": "Tit-for-Tat hoạt động như thế nào?",
  "collection": "ly-thuyet-tro-choi",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "answer": "Tit-for-Tat hoạt động theo nguyên tắc...",
  "sources": ["0:23–1:45", "3:12–4:30"],
  "facts": ["Tit-for-Tat → chiến thắng → cuộc thi"],
  "session_id": "uuid",
  "cached": false,
  "suggestions": ["Tại sao Tit-for-Tat thắng?", "Green Trigger là gì?", "Trò chơi tổng bằng không là gì?"]
}
```

**Rate limit:** 20 requests/minute per IP.

---

### `POST /chat/stream`

🔒 **Protected**

RAG chat (streaming). Returns `text/plain` stream.

Same request body as `/chat`. Stream format:
1. Answer text chunks streamed word-by-word
2. Final JSON meta frame: `\n\n__META__{...}`

**Meta frame schema:**
```json
{
  "sources": ["0:23–1:45"],
  "facts": ["entity → relation → entity"],
  "session_id": "uuid",
  "suggestions": ["question 1", "question 2", "question 3"],
  "cached": false
}
```

**Rate limit:** 20 requests/minute per IP.

---

### `GET /suggestions/{collection}`

🔒 **Protected**

Get 4 dynamic questions generated from video content.

**Response:**
```json
{"suggestions": ["Câu hỏi 1?", "Câu hỏi 2?", "Câu hỏi 3?", "Câu hỏi 4?"]}
```

**Rate limit:** 10 requests/minute per IP.

---

### `GET /summarize/{collection}`

🔒 **Protected**

Get full video summary (synchronous).

```json
{"summary": "Video nói về lý thuyết trò chơi..."}
```

---

### `GET /summarize/stream/{collection}`

🔒 **Protected**

Get video summary (streaming). Returns `text/plain` stream.

---

### `GET /history/{session_id}?collection=...`

Get chat history for a session (last 20 messages).

```json
[
  {"role": "user", "content": "Tit-for-Tat là gì?"},
  {"role": "assistant", "content": "Tit-for-Tat là chiến thuật..."}
]
```

---

### `GET /quiz/{collection}`

🔒 **Protected**

Generate quiz (MCQ) or flashcards from video content.

**Query params:**

| Param | Default | Description |
|---|---|---|
| `mode` | `quiz` | `quiz` = MCQ trắc nghiệm, `flashcard` = Anki-style |
| `count` | `10` | Số câu hỏi / flashcard (1–20) |

**Response (mode=quiz):**
```json
{
  "mode": "quiz",
  "collection": "ly-thuyet-tro-choi",
  "items": [
    {
      "question": "Tit-for-Tat hoạt động như thế nào?",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "A",
      "explanation": "Tit-for-Tat bắt đầu bằng hợp tác...",
      "timestamp": "3:12",
      "start_time": 192.0
    }
  ]
}
```

**Response (mode=flashcard):**
```json
{
  "mode": "flashcard",
  "collection": "ly-thuyet-tro-choi",
  "items": [
    {
      "front": "Tit-for-Tat",
      "back": "Chiến thuật: bắt đầu hợp tác, sau đó phản chiếu hành động của đối thủ.",
      "timestamp": "3:12",
      "start_time": 192.0
    }
  ]
}
```

**Rate limit:** 10 requests/minute per IP.

---

### `POST /graph/build/{collection}`

🔒 **Protected**

Build or rebuild Knowledge Graph for a collection.

```json
{"status": "success", "collection": "ly-thuyet-tro-choi", "nodes": 130, "edges": 86}
```

---

### `DELETE /collections/{name}`

🔒 **Protected**

Delete a video collection (Qdrant + graph + SPLADE cache).

```json
{"status": "deleted", "collection": "ly-thuyet-tro-choi"}
```

---

## Error Codes

| Code | Meaning |
|---|---|
| 401 | Invalid or missing `X-API-Key` |
| 422 | Invalid YouTube URL |
| 429 | Rate limit exceeded |
| 503 | Service misconfigured (whitespace API key) |
