# YouRAG: Nền tảng Tìm kiếm Video YouTube & Tri thức Tích hợp AI (SOTA)

**YouRAG** là một hệ thống RAG (Retrieval-Augmented Generation) hiện đại, lấy luồng đầu vào là các video YouTube, "hấp thụ" (ingest) thông tin từ transcript/metadata, lập chỉ mục đa chiều và cho phép **Tra cứu Thông minh (Hybrid Search)** cũng như **Tóm tắt (Summarization)** dựa trên sức mạnh của các Large Language Model (LLM) hàng đầu thế giới (Groq, Llama, BAAI/bge).

Thay vì phải xem trọn vẹn đoạn video dài, YouRAG cho phép bạn bóc tách sâu kiến thức, hỏi đáp logic và nhanh chóng lấy được chính xác mốc thời gian (timestamp) của video tương ứng với câu hỏi của bạn.

---

## 🚀 Tính năng Cốt lõi (Core Features)

### 1. MLOps & Ingestion SOTA (Hấp thụ và Đóng gói Kiến thức)

- **ZenML Orchestration**: Luồng xử lý dữ liệu (Pipeline) được quản trị và điều phối hoàn toàn bằng **ZenML**, mang lại khả năng theo dõi tiến trình (Tracking) trực quan qua giao diện đồ thị DAG, giúp hệ thống chịu lỗi xuất sắc, không bao giờ phải cào lại dữ liệu từ đầu nếu mất mạng.
- **Tải Metadata & Phụ đề song song**: Xử lý tải dữ liệu thô cực nhanh, lọc ký tự thừa chuẩn xác.
- **Tiến trình Semantic Chunking**: Bóc lẻ nội dung dựa trên vector ngữ nghĩa kết hợp thuật toán Binary Search `O(log n)` để định vị siêu tốc thời gian (Start/End time) cho mỗi khối chữ (Chunk).
- **Contextual Enrichment (Cải tiến Anthropic)**: Bơm thêm 1-2 câu ngữ cảnh gốc do LLM Llama3 suy luận vào đầu mỗi Chunk tĩnh, tiêu diệt hoàn toàn tình trạng "Lost in the middle" ở RAG cổ điển.
- **Graph & Entity Extraction**: Tự động trích xuất Noun Phrases, Abbreviation bằng Rule-based & Regex làm siêu dữ liệu (Metadata) siêu nhạy cho Sparse Search.

### 2. Retrieval (Tra cứu Thông minh)

Áp dụng phương pháp **Hybrid RAG** (Tìm kiếm Kép) kết hợp **Reranker** khắt khe nhất:

- **Dense Search (Tìm Vector)**: Nhúng nhãn câu hỏi qua `BAAI/bge-m3` đa ngôn ngữ siêu việt.
- **Sparse Search (BM25)**: Bắt dính tuyệt đối những từ khóa chuyên ngành, mã hiệu sản phẩm (Exact-match).
- **Reciprocal Rank Fusion (RRF)**: Trộn kết quả và dung hòa điểm số (Fusion Rank) từ cả 2 mảng trên.
- **Cross-Encoder Reranking**: Dùng trạm kiểm soát `mmarco-mMiniLMv2` chấm điểm chéo độc lập từng cặp (Query, Chunk) để lọc rác cực đoan (Zero-noise) trước khi giao cho LLM trả lời.

### 3. Sáng tạo và Tóm tắt (Text Generation)

- **Video Summarization (Tóm tắt video)**: Với 1 click, AI Llama3 70B sẽ đọc lại chuỗi Chunk của ChromaDB, ghi nhận thời gian tự nhiên và thay bạn tường thuật lại toàn cảnh nội dung Video.
- **Streaming Response**: Chữ nhảy ra từ từ giống hệt ChatGPT kèm theo Nguồn (Timestamp) nhúng link HTML bấm mở thẳng Video Youtube để kiểm chứng.

---

## 🛠 Cấu trúc Công nghệ (Tech Stack)

| Lớp (Layer)            | Công nghệ sử dụng                    | Chức năng (Role)                             |
| ---------------------- | ------------------------------------ | -------------------------------------------- |
| **Orchestration MLOps**| `ZenML`                              | Quản trị/Điều phối luồng Data Pipeline       |
| **Embedding**          | `BAAI/bge-m3`                        | Nhúng Vector ngữ nghĩa mạnh mẽ               |
| **Reranker**           | `mmarco-mMiniLMv2-L12-H384`          | Mô hình chấm điểm và lọc nhiễu Retrieval     |
| **Vector DB**          | `ChromaDB`                           | Lưu trữ Vector và phân chia Query            |
| **Contextual LLM**     | `llama-3.1-8b-instant` (qua Groq)    | Phân tích và sinh bộ ngữ cảnh bơm vào Chunk  |
| **Generation LLM**     | `llama-3.3-70b-versatile` (qua Groq) | Bộ não sinh từ và tương tác với User         |
| **Frontend/Backend**   | `Streamlit` / `FastAPI`              | Giao diện Chat UI & Máy chủ API              |

---

## ⚙ Tiền Quyết & Thiết Lập Hệ Thống

**1. Clone mã nguồn và Cài đặt Environment**
```bash
poetry install
```

**2. Cấu hình Biến Môi trường (.env)**
Trong file `.env` root, cung cấp API của Groq:
```env
# SECRETS
GROQ_API_KEY="gsk_xxxxx"

# MODELS SETUP
EMBEDDING_MODEL_NAME="BAAI/bge-m3"                  
CROSS_ENCODER_MODEL="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"       
LLM_MODEL_NAME="llama-3.3-70b-versatile"             
LLM_CONTEXTUAL_MODEL="llama-3.1-8b-instant"          
LLM_PROVIDER="groq"                                
```

**3. Khởi chạy Hệ Sinh Thái MLOps (Bấm 3 Terminal cúng lúc)**
```bash
# Terminal 1: Bật máy chủ FastAPI đằng sau
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Bật trang Web giao diện UI
poetry run streamlit run app.py

# Terminal 3: Bật trang theo dõi luồng Data Pipeline của ZenML
poetry run zenml up
```

---

## 🗺 Lộ Trình Phát Triển (Roadmap - ĐÃ HOÀN TẤT 100%)

Dự án hiện tại đã vươn tới trạm cuối cùng của kiến trúc MLOps RAG thực chiến!

### Phase 1: Ingestion Pipeline (Done)
- [x] Chunking Timestamp Mapping siêu tốc
- [x] Contextual Enrichment SOTA
- [x] Graph Keyword Extraction Rule-based
- [x] ZenML MLOps Orchestration Integration

### Phase 2: Hybrid Retrieval & Generation (Done)
- [x] BM25 Sparse Search
- [x] Dense Search
- [x] RRF Hybrid Search
- [x] Video Summarization Moduler
- [x] Cross-Encoder Reranking
- [x] Prompt Builder Module

### Phase 3: Giao Diện Tương Tác & API (Done)
- [x] FastAPI Server cung cấp Streaming Chunk Data
- [x] Streamlit Chat UI siêu mượt với Dark/Glassmorphism theme
- [x] Clickable Youtube Timestamp Citation HTML Tags

---

## ✅ Những Nâng Cấp Tương Lai (Gap to pure Production)
Các hạng mục dưới đây giúp YouRAG đi từ bản MLOps cá nhân sang bản thương mại hóa cho vạn User:
- Khắc phục Single-Point-of-Failure: Bổ sung LLM Router (Fallback từ Groq sang OpenAI/Gemini nếu cháy Rate Limit).
- Đóng gói (Containerize): Bọc gesamte hệ thống vào các file Docker Compose để dễ dàng Scale in Kubernetes.
- Theo dõi Hành vi (Tracking): Tích hợp Arize Phoenix hoặc LangSmith để đo lượng Token hao hụt và bắt "vết" LLM Hallucinations.
- Vector DB: Nâng cấp thư mục ChromaDB tĩnh bằng Máy chủ Milvus dã chiến trên Cloud.

> Được thiết kế với tư duy Software Architecture vững chắc. Sẵn sàng tích hợp cho các Nền tảng AI Thương mại.
