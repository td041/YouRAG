from typing import Optional, Literal
from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env fallback cho môi trường thư mục local
load_dotenv(override=True)

class Settings(BaseSettings):
    """Lõi Cấu hình Đẳng Cấp Production v2.0
    - Tự động bắt biến môi trường (OS Env) hoặc từ file .env
    - Phân chia môi trường rõ ràng (dev/prod).
    - Bảo mật tuyệt đối Key bằng SecretStr (chống rủi ro in leak key ra Log).
    """
    # 1. Định Tinh Ứng Dụng (App Core)
    PROJECT_NAME: str = "YouRAG Production"
    VERSION: str = "1.1.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # 2. Quản Lý Chìa Khóa (Secrets Vault) - Ẩn giấu khi Print
    API_KEY: Optional[SecretStr] = None  # Bearer token bảo vệ các endpoint write
    OPENAI_API_KEY: Optional[SecretStr] = None
    GROQ_API_KEY: Optional[SecretStr] = None
    GROQ_API_KEYS: str = ""  # Danh sách key dự phòng, cách nhau bởi dấu phẩy: "key1,key2,key3"
    MISTRAL_EVAL_API_KEY: Optional[SecretStr] = None  # RAGAS benchmark evaluator (no daily quota)
    JINA_API_KEY: Optional[SecretStr] = None  # Late Chunking (jina-embeddings-v3)
    GEMINI_API_KEY: Optional[SecretStr] = None  # Visual Frame RAG (Gemini 1.5 Flash vision)

    # 3. Vector Database Engine (QDRANT VƯƠNG GIẢ)
    QDRANT_DB_PATH: str = "qdrant_db"
    QDRANT_SERVER_URL: Optional[str] = None # Khi lên Prod, điền URL vào biến này
    QDRANT_API_KEY: Optional[SecretStr] = None

    # 4. LLM & Nhúng Vector (Brain Models)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"           
    CROSS_ENCODER_MODEL: str = "BAAI/bge-reranker-v2-m3"  # SOTA multilingual reranker (tiếng Việt tốt hơn mmarco)
    LLM_MODEL_NAME: str = "llama-3.3-70b-versatile"     
    LLM_CONTEXTUAL_MODEL: str = "llama-3.1-8b-instant"  
    LLM_PROVIDER: Literal["groq", "openai", "ollama"] = "groq"  # gemini removed
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"

    # 5. Cấu hình vận hành (System Dials)
    DEVICE: str = "cuda" # 'cuda' hoặc 'cpu'
    LOG_LEVEL: str = "INFO"
    VECTOR_SIZE: int = 1024 # BGE-M3
    
    # 6. Các Núm Xoay Động Cơ RAG (RAG Tuning Dials)
    TOP_K_RETRIEVAL: int = 15
    TOP_K_RERANK: int = 3
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    SUMMARIZER_MAX_CHUNKS: int = 12   # Số chunks tối đa để summarize (Groq TPM limit)
    GRAPH_MAX_CHUNKS: int = 40        # Số chunks tối đa để build knowledge graph (video dài cần nhiều hơn)
    YOUTUBE_FETCH_TIMEOUT: int = 120  # Timeout (seconds) cho YouTube metadata + transcript fetch

    # 7. Cấu hình Kết nối (Connection Hub)
    # Tự động lấy từ Docker Compose hoặc file .env
    DATABASE_URL: str = "sqlite:///./qdrant_db/chat_history.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # 8. Observability — Langfuse LLM Tracing (optional)
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[SecretStr] = None
    LANGFUSE_HOST: str = "http://langfuse:3000"

    # Pydantic v2 chuẩn:
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # Ignore các biến dư thừa trong .env để tránh rác RAM
    )
    
    @computed_field
    def is_production(self) -> bool:
        """Thuộc tính đọc nhanh để các hàm khác tự Tắt/Bật tính năng xịn."""
        return self.ENVIRONMENT == "production"

# Singleton Design Pattern (Only 1 instance in RAM)
settings = Settings()
