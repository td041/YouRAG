import logging
import sys

def setup_logger(name: str = "YouRAG", level=logging.INFO):
    """Cấu hình Log chuẩn Production hiển thị JSON (ẩn trong ví dụ này để dễ nhìn)"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger()
