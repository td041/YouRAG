import time
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.exc import OperationalError
from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger("YouRAG_Postgres")

# Sử dụng DATABASE_URL từ centralized settings
DATABASE_URL = settings.DATABASE_URL

# Nếu dùng SQLite thì cần argument check_same_thread=False
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)

def init_db(retries: int = 5, delay: int = 5):
    """Tạo các bảng dữ liệu nếu chưa tồn tại, có cơ chế thử lại nếu DB chưa sẵn sàng"""
    logger.info(f"Khởi tạo cấu trúc bảng PostgreSQL tại {DATABASE_URL}...")
    
    for i in range(retries):
        try:
            SQLModel.metadata.create_all(engine)
            logger.info("✅ Đã khởi tạo xong các bảng PostgreSQL.")
            return
        except OperationalError as e:
            if i < retries - 1:
                logger.warning(f"⚠️ Chưa kết nối được DB (Lần {i+1}/{retries}). Thử lại sau {delay}s... Error: {e}")
                time.sleep(delay)
            else:
                logger.error(f"❌ Không thể kết nối tới PostgreSQL sau {retries} lần thử. Lỗi: {e}")
                raise

def get_session():
    """Generator để lấy database session dùng cho FastAPI Depends"""
    with Session(engine) as session:
        yield session
