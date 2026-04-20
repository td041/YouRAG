class SemanticCache:
    """Caching sử dụng Embedding để đối chiếu độ tương quan của câu hỏi."""
    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
        self.cache_db = {} # TODO: Sử dụng Redis / SQLite thực tế

    def check_cache(self, query: str) -> str | None:
        """Nhúng (Embed) query và tìm kiếm câu trả lời đã tồn tại."""
        # 1. Embed query
        # 2. Vector search trong Cache DB với threshold > 0.95
        # 3. Trả về answer nếu Hit Cache
        return None

    def save_to_cache(self, query: str, answer: str):
        """Lưu lại kết quả câu trả lời để tiết kiệm query lần sau."""
        # TODO: Lưu (query_vector, answer) vào bộ đệm
        pass
