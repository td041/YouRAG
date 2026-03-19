import logging
import sys
from .config import settings

def setup_logger(name: str = "YouRAG"):
    """Cấu hình Log chuẩn Production"""
    # Lấy level từ settings (mặc định INFO)
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Tránh duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger()
