# YouRAG: Nền tảng Tìm kiếm Video YouTube & Tri thức Tích hợp AI (Advanced SOTA)

**YouRAG** là một hệ thống RAG (Retrieval-Augmented Generation) hiện đại bậc nhất, lấy luồng đầu vào là các video YouTube, "hấp thụ" (ingest) thông tin kết hợp **Đồ thị tri thức (Knowledge Graph)** nhằm cung cấp khả năng hỏi đáp và phân tích sâu mức độ chuyên gia. 

Thay vì sử dụng RAG truyền thống (chặt văn bản và tìm kiếm thô), YouRAG áp dụng **Ablation-tested SOTA Pipeline**, từ Ingestion đến Evaluation, nhằm tối ưu hóa độ chính xác và loại bỏ triệt để tình trạng ảo giác (Hallucination) của AI.

---

## 🚀 Tính năng Cốt lõi (Advanced Core Features)

### 1. 🏗️ MLOps & Ingestion (Hấp thụ và Đóng gói Kiến thức)
- **ZenML Orchestration**: Luồng xử lý dữ liệu được quản trị hoàn toàn vòng đời bằng MLOps pipeline.
- **Semantic Chunking & Contextual Enrichment**: Áp dụng kỹ thuật Anthropic, bơm thêm 1-2 câu ngữ cảnh gốc do LLM suy luận vào đầu mỗi Chunk để chống lại hiện tượng "Lost in the middle".
- **LLM-based Graph Extraction**: Tự động sử dụng LLM để đọc, trích xuất bộ ba (Thực thể - Quan hệ - Đối tượng) từ video, dọn đường cho Graph RAG.

### 2. 🔍 Retrieval (Tra cứu Thông minh màng lọc 3 lớp)
- **Qdrant Vector Engine**: Thay thế hoàn toàn ChromaDB bằng Qdrant Local Engine (Rust-based) cực nhẹ và siêu tốc, dùng `BAAI/bge-m3` nhúng Vector.
- **NetworkX Graph Traversal**: Kích hoạt **Graph RAG** tìm kiếm các liên kết ẩn (multi-hop reasoning) mà Vector Search bị mù.
- **Hybrid Search + RRF**: Tìm kiếm kép kết hợp Vector (Ý nghĩa) và BM25 (Từ khóa), hòa trộn điểm bằng thuật toán Reciprocal Rank Fusion.
- **Cross-Encoder Reranking**: Trạm kiểm soát `mmarco-mMiniLMv2` chấm điểm chéo độc lập để lọc rác cực đoan (Zero-noise) trước khi giao cho LLM.

### 3. 🧠 Generation (Bộ não tự soát lỗi)
- **Graph-based Self-Correction**: Tính năng SOTA đỉnh cao! AI bắt buộc phải đối chiếu bản nháp câu trả lời với "Graph Facts" (Sự thật đồ thị). Nếu phát hiện mâu thuẫn hoặc ảo giác, AI phải tự động sửa lỗi trước khi xuất ra cho người dùng.
- **Dynamic PromptBuilder**: Hỗ trợ xuất định dạng linh hoạt: Tự động vẽ sơ đồ tư duy (Mindmap qua Mermaid) hoặc kẻ bảng So sánh (Markdown Tables).
- **Streaming Response**: Gõ chữ mượt mà kết hợp trích dẫn mốc thời gian `[mm:ss]` (bấm để xem thẳng video Youtube).

### 4. 🧪 Evaluation (Kiểm định Chất lượng)
- **RAGAS Ablation Study**: Đường ống Benchmark tự động đo đạc sự tiến bộ của từng lớp SOTA (Naive → Hybrid → Reranker) qua 4 chỉ số vàng: *Faithfulness, Answer Relevancy, Context Precision, Context Recall*.

---

## 🛠 Cấu trúc Công nghệ (Tech Stack)

| Lớp (Layer)            | Công nghệ sử dụng                    | Chức năng (Role)                             |
| ---------------------- | ------------------------------------ | -------------------------------------------- |
| **Orchestration**      | `ZenML`                              | Quản trị/Điều phối luồng Data Pipeline       |
| **Core & Security**    | `Pydantic v2`                        | Quản lý cấu hình `SecretStr`, Singleton DB   |
| **Vector DB**          | `Qdrant`                             | Máy chủ tìm kiếm Vector siêu tốc             |
| **Graph DB**           | `NetworkX`                           | Lưu trữ và tính toán đường đi Đồ thị         |
| **Embedding & Rerank** | `BAAI/bge-m3` & `mmarco-mMiniLMv2`   | Mô hình nhúng Vector và chấm điểm nhiễu      |
| **Evaluation**         | `RAGAS Framework`                    | LLM-as-a-judge chấm điểm tự động             |
| **LLM Provider**       | `llama-3.3-70b` (qua Groq API)       | Trái tim tạo ngôn ngữ và tự soát lỗi (Self-RAG)|

---

## ⚙ Thiết Lập Hệ Thống

**1. Clone mã nguồn và Cài đặt Environment**
```bash
poetry install
```

**2. Cấu hình Biến Môi trường (.env)**
```env
# SECRETS
GROQ_API_KEY="gsk_xxxxx"

# SYSTEM DIALS
DEVICE="cuda" # hoặc "cpu"
LOG_LEVEL="INFO"
```

**3. Khởi chạy Hệ Sinh Thái Benchmark (Kiểm chứng SOTA)**
```bash
poetry run python tests/run_benchmark.py --evaluate
```

---

## ✅ GAP TO PURE PRODUCTION (Những gì cần làm tiếp theo)
Hệ thống hiện tại đã đạt cấu trúc lõi rất mạnh, tuy nhiên để thực sự **"Production-ready"** và chịu tải cho ngàn User, dự án cần triển khai các hệ thống sau:

1. **Semantic Caching (Redis/GPTCache)**:
   - *Vấn đề:* Gọi LLM cho các câu hỏi trùng lặp rất tốn kém và chậm.
   - *Giải pháp:* Lắp bộ nhớ đệm Vector. Nếu câu hỏi mới trùng 95% ý nghĩa với câu hỏi cũ, móc ngay câu trả lời trong Cache ra (độ trễ 0.1s, chi phí $0).
2. **Semantic Router & API (FastAPI)**:
   - Dựng máy chủ RESTful API. Áp dụng Semantic Router để phân loại câu hỏi (Ví dụ: Hỏi thời tiết -> Chặn; Hỏi đếm số lượng -> Chuyển thẳng về Graph DB không qua Vector DB).
3. **Frontend Application (Streamlit/Next.js)**:
   - Xây dựng Web UI có chế độ Darkmode/Glassmorphism. Hiển thị trực quan Mindmap Mermaid bằng React component.
4. **Containerization & CI/CD (Docker + GitHub Actions)**:
   - Cần đóng hộp (Dockerize) toàn bộ App, Qdrant và Redis vào `docker-compose.yml`.
   - Thiết lập GitHub Actions tự động Lint code (`ruff`), test RAGAS, và đẩy Image lên GHCR mỗi khi có Push.
5. **Observability (Phoenix/LangSmith)**:
   - Tích hợp công cụ đo vết (Tracing) để xem mỗi câu trả lời ngốn bao nhiêu Token và tốn bao nhiêu giây ở từng bước.
