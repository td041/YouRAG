"""
Dataset Generator cho RAGAS Evaluation.

Tự động tạo bộ câu hỏi đánh giá (Evaluation Dataset) từ video đã Ingest.
Dùng LLM để sinh câu hỏi + đáp án từ các Chunks thực tế trong Qdrant.
"""

import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.database import db_instance
from src.core.logger import logger
from src.core.utils import format_timestamp
from src.engine.generation.llm_client import LLMClient


class EvalDatasetGenerator:
    """Tạo bộ Evaluation Dataset từ dữ liệu video đã Ingest.
    
    Cách hoạt động:
    1. Kéo các Chunks từ Qdrant (đã được Ingestion trước đó)
    2. Trích mẫu đều (Uniform Sampling) để bao phủ toàn bộ video
    3. Dùng LLM sinh câu hỏi + đáp án (Ground Truth) cho mỗi chunk
    4. Xuất ra file JSON chuẩn RAGAS format
    """

    def __init__(self, num_questions: int = 10):
        self.llm = LLMClient()
        self.db = db_instance
        self.num_questions = num_questions

    def _fetch_chunks(self, collection_name: str) -> List[Dict[str, Any]]:
        """Kéo toàn bộ chunks từ Qdrant và sắp xếp theo thời gian."""
        records = []
        offset = None
        while True:
            result, next_offset = self.db.client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            records.extend(result)
            if next_offset is None:
                break
            offset = next_offset

        chunks = []
        for rec in records:
            payload = rec.payload or {}
            raw_text = payload.get("text", "")
            # Bóc tách phần context enrichment nếu có
            if "\n\n" in raw_text:
                raw_text = raw_text.split("\n\n")[-1]

            chunks.append({
                "id": rec.id,
                "content": raw_text,
                "index": payload.get("chunk_index", 0),
                "start_time": payload.get("start_time", 0.0),
                "end_time": payload.get("end_time", 0.0),
            })

        chunks.sort(key=lambda x: x["index"])
        return chunks

    def _sample_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Trích mẫu đều (Uniform Sampling) để bao phủ toàn bộ video."""
        if len(chunks) <= self.num_questions:
            return chunks

        step = len(chunks) / self.num_questions
        sampled = [chunks[int(i * step)] for i in range(self.num_questions)]
        return sampled

    def _generate_qa_from_chunk(self, chunk: Dict, chunk_idx: int) -> Dict:
        """Dùng LLM sinh 1 cặp Question + Answer từ 1 chunk."""
        ts = format_timestamp(chunk["start_time"])

        prompt = f"""Dựa trên đoạn transcript video sau đây (ở mốc thời gian [{ts}]):

<chunk>
{chunk['content']}
</chunk>

Hãy tạo ra CHÍNH XÁC 1 câu hỏi và 1 câu trả lời dựa trên nội dung trên.
Yêu cầu:
- Câu hỏi phải CỤ THỂ, có thể trả lời được từ đoạn text trên.
- Câu trả lời phải CHÍNH XÁC, chỉ dựa vào thông tin trong đoạn text.
- Trả lời bằng Tiếng Việt.

Trả về đúng format JSON (KHÔNG thêm markdown, KHÔNG thêm ```json):
{{"question": "câu hỏi", "ground_truth": "câu trả lời chi tiết"}}"""

        try:
            response = self.llm.chat_complete(
                prompt=prompt,
                system="Bạn là chuyên gia tạo bộ đề thi đánh giá hệ thống AI. Trả về JSON thuần, không markdown.",
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse JSON từ response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            qa = json.loads(response)
            qa["start_time"] = chunk["start_time"]
            qa["end_time"] = chunk["end_time"]
            qa["source_chunk_index"] = chunk["index"]
            qa["reference_context"] = chunk["content"]
            
            logger.info(f"   ✅ [{chunk_idx}] Q: {qa['question'][:60]}...")
            return qa

        except Exception as e:
            logger.error(f"   ❌ [{chunk_idx}] Lỗi sinh QA: {e}")
            return {
                "question": f"Nội dung video ở mốc {ts} nói về điều gì?",
                "ground_truth": chunk["content"][:200],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "source_chunk_index": chunk["index"],
                "reference_context": chunk["content"],
            }

    def generate(self, collection_name: str, output_path: str = None) -> List[Dict]:
        """Pipeline chính: Tạo Evaluation Dataset từ video đã Ingest.
        
        Args:
            collection_name: Tên Collection trong Qdrant
            output_path: Đường dẫn xuất file JSON (mặc định: tests/benchmark/eval_dataset.json)
            
        Returns:
            List các dict chứa question, ground_truth, context
        """
        logger.info(f"🧪 RAGAS Dataset Generator: Bắt đầu tạo {self.num_questions} câu hỏi từ [{collection_name}]")
        
        # 1. Kéo chunks
        all_chunks = self._fetch_chunks(collection_name)
        if not all_chunks:
            logger.error("❌ Collection rỗng!")
            return []
        logger.info(f"   📦 Tổng chunks: {len(all_chunks)}")
        
        # 2. Trích mẫu đều
        sampled = self._sample_chunks(all_chunks)
        logger.info(f"   🎯 Đã trích mẫu: {len(sampled)} chunks")
        
        # 3. Sinh QA cho từng chunk
        dataset = []
        for idx, chunk in enumerate(sampled, 1):
            logger.info(f"\n   📝 Đang sinh câu hỏi {idx}/{len(sampled)}...")
            qa = self._generate_qa_from_chunk(chunk, idx)
            dataset.append(qa)
        
        # 4. Xuất file
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "tests", "benchmark", "eval_dataset.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n🎉 Đã tạo xong {len(dataset)} câu hỏi → {output_path}")
        return dataset
