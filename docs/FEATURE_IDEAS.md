# Feature Ideas for YouRAG

Tai lieu nay tong hop cac tinh nang mo rong ban co the them vao YouRAG theo nhom uu tien.

## 1. Web Grounding from Video Context (de xuat quan trong)

Muc tieu: khi nguoi dung hoi ve mot noi dung trong video, he thong co the tim them nguon web lien quan de doi chieu va bo sung bang chung.

### Luong de xuat

1. User query + top chunks tu video.
2. Query expansion: tao 2-5 truy van tim kiem tu keywords/entity trong chunk.
3. Web search (news/docs/blog/chinh chu) voi bo loc domain uy tin.
4. Crawl + extract main content.
5. Rerank theo do lien quan voi query va noi dung video.
6. Tra ve cau tra loi co hai nhom trich dan:

- Video citations ([mm:ss])
- Web citations (title, url, snippet)

### Kien truc goi y

- Module moi: `src/engine/retrieval/web_grounding.py`
- Schema moi: `src/schema/web_sources.py`
- API moi:
  - `POST /chat/grounded`
  - `POST /chat/grounded/stream`
- Config moi trong `.env`:
  - `WEB_SEARCH_PROVIDER=serpapi|tavily|searxng`
  - `WEB_GROUNDING_TOP_K=5`
  - `TRUSTED_DOMAINS=...`

### Quy tac chat luong

- Neu web source mau thuan voi video: danh dau "conflict".
- Uu tien nguon chinh chu (docs, publisher, nha san xuat).
- Loai bo trang SEO spam, content farm.

## 2. Multi-Video Knowledge Base

Muc tieu: hoi dap tren nhieu video cung chu de, khong gioi han 1 collection.

De xuat:

- Them endpoint `POST /chat/multi` nhan danh sach collections.
- Them bo loc theo channel/date/topic.
- UI cho phep tick nhieu video truoc khi dat cau hoi.

## 3. Fact-Check Mode

Muc tieu: danh gia muc do tin cay cua cau tra loi.

De xuat output:

- `confidence_score`
- `evidence_count`
- `conflict_flags`
- `missing_evidence` (neu cau hoi vuot qua du lieu)

## 4. Semantic Cache thuc chien

Muc tieu: giam token va latency cho cac cau hoi giong nhau.

De xuat:

- Luu `query_embedding`, `normalized_answer`, `sources`.
- Match theo cosine >= threshold.
- TTL cache + cache invalidation theo collection version.

## 5. Auto Evaluation Pipeline

Muc tieu: do chat luong RAG theo thoi gian.

De xuat chi so:

- Retrieval hit rate@k
- Answer groundedness
- Citation correctness
- Latency p50/p95

Cong cu goi y: RAGAS, custom benchmark set tu transcript that.

## 6. Better Ingestion for Real-world Videos

De xuat:

- Ho tro video khong co subtitle: fallback ASR (Whisper).
- Speaker segmentation cho podcast/interview.
- Chapter detection + topic segmentation.
- Language detection va dich subtitle neu can.

## 7. Product UX upgrades

De xuat:

- Muc "Why this answer" hien thi retrieval path.
- Timeline navigator: click vao section summary de nhay video.
- Export markdown report (answer + citations + links).
- Saved queries and session history theo collection.

## 8. Security and Operations

De xuat:

- JWT/API key auth cho API.
- Per-user quota va rate limit.
- Audit log cho ingestion/chat actions.
- Docker + CI/CD + staging environment.

## Suggested implementation order

1. Summary cache + API validation
2. Semantic cache
3. Web grounding
4. Multi-video chat
5. Fact-check mode
6. Evaluation pipeline
