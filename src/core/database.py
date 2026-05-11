from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from .config import settings
from .logger import logger
import os
import torch

class VectorDatabase:
    """Singleton Qdrant Client siêu tốc & cực nhẹ (Rust-based).
    
    Qdrant hỗ trợ tính năng File-based chạy thẳng trên Local cực kỳ nhẹ,
    mà không cần đến Server Docker phức tạp như Milvus hay Qdrant Server. 
    Khi lên Production, chỉ cần đổi `path` thành `url` là hệ thống scale 10,000 requests.
    """
    _instance = None
    _client = None
    _embedding_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDatabase, cls).__new__(cls)

            # 1. Khởi tạo Qdrant Client (Hỗ trợ Local & Server Mode)
            if settings.QDRANT_SERVER_URL:
                api_key = settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
                cls._client = QdrantClient(url=settings.QDRANT_SERVER_URL, api_key=api_key)
                logger.info(f"🌐 Qdrant Server Engine khởi động tại: {settings.QDRANT_SERVER_URL}")
            else:
                db_path = os.path.abspath(settings.QDRANT_DB_PATH)
                os.makedirs(db_path, exist_ok=True)
                cls._client = QdrantClient(path=db_path)
                logger.info(f"⚡ Qdrant Local Engine khởi động tại: {db_path}")

            # 2. Tải Mô hình Vector
            device = settings.DEVICE if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Đang nạp SentenceTransformer: {settings.EMBEDDING_MODEL_NAME} (Device: {device})...")
            cls._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=device)
            logger.info("✅ Qdrant Embedding Model Loaded!")

        return cls._instance

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def embedding_model(self) -> SentenceTransformer:
        return self._embedding_model

    def get_or_create_collection(self, name: str):
        """Khởi tạo Bảng chứa dữ liệu (Collection)."""
        # Kiểm tra xem bộ chia đã tồn tại chưa
        if not self._client.collection_exists(collection_name=name):
            # Tạo mới với độ dài vector chuẩn
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"🆕 Đã tạo mới Qdrant Collection: {name}")
            
        return name

# Cổng truy xuất Toàn Cục (Singleton)
db_instance = VectorDatabase()
