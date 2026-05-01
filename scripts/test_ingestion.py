import sys
import os

# Thêm đường dẫn project vào sys.path để có thể import từ src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.ingestion.pipeline import IngestionPipeline
from src.core.logger import logger
from src.core.database import db_instance

def main():
    # Video 3Blue1Brown giải thích Neural Network (chắc chắn có phụ đề tiếng Anh vietsub)
    test_url = "https://www.youtube.com/watch?v=bQ7EHpVHUMs"

    try:
        logger.info("=== BẮT ĐẦU TEST SCRIPT INGESTION SOTA ===")
        
        # Khởi tạo Pipeline
        pipeline = IngestionPipeline()
        
        # Chạy nạp dữ liệu vào DB
        result = pipeline.run(test_url)
        print("\n\n")
        
        logger.info(f"🟢 [THÀNH CÔNG] Pipeline Ingestion chạy xong mà không bị sập.")
        logger.info(f"Video đã xử lý: '{result['title']}'")
        logger.info(f"Số lượng Chunks ngữ nghĩa được tạo & đưa vào DB: {result['chunks_added']}")
        
        # Kiểm tra CSDL thực tế
        collection = db_instance.get_collection("youtube_knowledge")
        db_count = collection.count()
        logger.info(f"📦 Tổng số lượng Chunks hiện có trong Qdrant 'youtube_knowledge': {db_count}")
        
    except Exception as e:
        logger.error(f"🔴 ERROR: Quá trình Test Ingestion thất bại. Lý do: {str(e)}")

if __name__ == "__main__":
    main()
