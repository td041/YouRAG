# YouRAG: Nền tảng Tìm kiếm Video YouTube & Tri thức Tích hợp AI (SOTA)

**YouRAG** là một hệ thống RAG (Retrieval-Augmented Generation) hiện đại, lấy luồng đầu vào là các video YouTube, "hấp thụ" (ingest) thông tin từ transcript/metadata, lập chỉ mục đa chiều và cho phép **Tra cứu Thông minh (Hybrid Search)** cũng như **Tóm tắt (Summarization)** dựa trên sức mạnh của các Large Language Model (LLM) hàng đầu thế giới (Groq, Llama, BAAI/bge).

Thay vì phải xem trọn vẹn đoạn video dài, YouRAG cho phép bạn bóc tách sâu kiến thức, hỏi đáp logic và nhanh chóng lấy được chính xác mốc thời gian (timestamp) của video tương ứng với câu hỏi của bạn.

---

## 🚀 Tính năng Cốt lõi (Core Features)

### 1. Ingestion SOTA (Hấp thụ và Đóng gói Kiến thức)
- **Tải Metadata & Phụ đề song song**: Xử lý tải dữ liệu thô cực nhanh, tự động lọc `\n` và những ký tự UTF-8 thừa.
- **Tiến trình Sematic Chunking**: Bóc lẻ nội dung phụ đề dựa trên logic độ dài câu, đồng thời sử dụng Binary Search `O(log n)` để định vị siêu tốc thời gian (Start/End time) cho mỗi khối chữ (Chunk).
- **Contextual Enrichment (Cải tiến của Anthropic - Phân đoạn ngữ cảnh SOTA)**: Mỗi Chunk chữ được bơm thêm 1-2 câu ngữ cảnh gốc do LLM Llama3 suy luận và phác thảo, giảm thiểu tình trạng "mất não" ở RAG cổ điển. Tính năng có hỗ trợ Cache để giảm độ trễ (latency).
- **Graph & Entity Extraction**: Tự động sử dụng cú pháp Regex và Rule-based (không tốn token LLM) để trích xuất những Noun Phrases, Abbreviation làm bộ Metadata siêu nhạy và rẻ tiền.
- **Vector Indexing độc lập**: Mỗi đoạn YouTube có 1 Collection (tiêu mục) riêng tại ChromaDB để hỗ trợ RAG đa luồng mà không lo nhiễu (noise) giữa các video khác biệt.

### 2. Retrieval (Tra cứu Thông minh)
Sức mạnh tuyệt đối của dự án nằm ở phần Truy xuất, áp dụng phương pháp **Hybrid Search** (Tìm kiếm kép):
- **Dense Search (Tìm Vector)**: Nhúng nhãn câu hỏi người dùng qua `BAAI/bge-m3` (đạt 1024 chiều, bao trọn Ngữ nghĩa).
- **Sparse Search (BM25)**: Nhấn mạnh vào từ khóa bắt buộc (như tên nhân vật "Harry Potter", mã hiệu "RTX 4090", từ lóng...).
- **Reciprocal Rank Fusion (RRF)**: Sự kết hợp hoàn hảo để xếp hạng (Fusion Rank) cho ra Top kết quả cao nhất từ cả 2 mảng trên.

### 3. Sáng tạo và Tóm tắt (Text Generation)
- **Video Summarization (Tóm tắt video)**: Với 1 click, AI Llama3 70B sẽ đọc lại chuỗi Chunk của ChromaDB, ghi nhận thời gian tự nhiên (thứ tự Timeline) và thay bạn tường thuật lại toàn cảnh nội dung Video, phân chia mạch lạc và gạch đầu dòng rõ ràng.

---

## 🛠 Cấu trúc Công nghệ (Tech Stack)

| Lớp (Layer) | Công nghệ sử dụng | Chức năng (Role) |
| --- | --- | --- |
| **Embedding** | `BAAI/bge-m3` | Embedding cục bộ cực mạnh mẽ |
| **Vector DB** | `ChromaDB` | Lưu trữ Vector và phân chia Query tốc độ cao |
| **Contextual LLM** | `llama-3.1-8b-instant` (qua Groq) | Đẻ ngữ cảnh chèn vào Chunk chữ |
| **Generation LLM** | `llama-3.3-70b-versatile` (qua Groq) | Bộ não cuối đưa ra câu trả lời/tóm tắt |
| **Sparse Engine** | `rank_bm25` | Engine so khớp từ khóa |
| **Cấu trúc Lập trình** | `Python (OOP, SOLID)` | Thiết kế Strategy/Facade Design Pattern |

---

## ⚙ Tiền Quyết & Thiết Lập Hệ Thống

**1. Clone mã nguồn và Cài đặt Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Cấu hình Biến Môi trường (.env)**
Trong file `.env` root, cung cấp API của bạn (hiện tại hỗ trợ Groq miễn phí):
```env
# SECRETS
GROQ_API_KEY="gsk_xxxxx"

# MODELS SETUP
EMBEDDING_MODEL_NAME="BAAI/bge-m3"
CROSS_ENCODER_MODEL="BAAI/bge-reranker-v2-m3"
LLM_MODEL_NAME="llama-3.3-70b-versatile"
LLM_CONTEXTUAL_MODEL="llama-3.1-8b-instant"
```

---

## 🗺 Lộ Trình Phát Triển (Roadmap)

Dự án hiện tại đã đạt độ bao quát tuyệt vời ở khâu Tiền xử lý (Ingestion) và Móc nối (Retrieval). Hành trình tiếp theo sẽ tối ưu thêm các Mảnh ghép Tiền tuyến (Front-End) và Reranker (Cắt rác cực đoan):

### Phase 1: Ingestion Pipeline `(Đã Hoàn Thành 100%)`
- [x] Chunking Timestamp Mapping siêu tốc
- [x] Contextual Enrichment SOTA
- [x] Graph Keyword Extraction Rule-based

### Phase 2: Hybrid Retrieval & Generation `(Đã Hoàn Thành 90%)`
- [x] BM25 Sparse Search
- [x] Dense Search
- [x] RRF Hybrid Search
- [x] Video Summarization Moduler 
- [ ] **Cross-Encoder Reranking**: Bước đệm lấy mảng Top N từ Hybrid gửi thẳng cho mô hình phán xử (Reranker) chấm lại điểm 0-1, lọc bay "Rác".
- [ ] **Prompt Builder Module**: Truyền tệp tin vào con RAG Generator để sinh ra câu trả lời hoàn hảo cho người Dùng.

### Phase 3: Xây dựng Giao Diện Tương Tác & API `(Chưa bắt đầu)`
- [ ] **FastAPI Server**: Cung cấp giao thức trả đổi `application/json` để kết nối Backend tới mọi hệ sinh thái.
- [ ] **Streamlit Tương tác (UI)**: Xây dựng giao diện chat Web. Người dùng dán link Youtube -> Máy Ingest -> Trả ra báo cáo tóm tắt trên Web -> Hiện khung Chat RAG tương tác cùng Video.

---

> Được xây dựng với tư duy Software Architecture (SOLID, SRP) nghiêm ngặt nhất. Sẵn sàng tích hợp cho các Nền tảng Sản xuất Thương mại (Production Environment).
