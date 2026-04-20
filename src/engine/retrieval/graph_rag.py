"""
Graph RAG - Truy xuất thông tin qua Đồ thị Tri thức (Knowledge Graph).

Kiến trúc:
1. Build Phase (Lúc Ingestion):
   - LLM trích xuất (Entity, Relationship, Entity) triples từ mỗi chunk
   - Xây dựng NetworkX Graph: Node = Entity, Edge = Relationship
   - Lưu Graph xuống disk (pickle) theo từng Collection

2. Query Phase (Lúc Chat):
   - Trích Entity từ câu hỏi user
   - Tìm Node khớp trong Graph → Duyệt các láng giềng (1-2 hop)
   - Thu thập tất cả chunk_ids liên quan → Kéo từ Qdrant
   - Kết hợp với kết quả Vector Search (Hybrid)

Ưu điểm so với RAG thông thường:
- Trả lời được câu hỏi QUAN HỆ: "Ai là cha của Eleven?"
- Trả lời được câu hỏi ĐẾM: "Có bao nhiêu nhân vật được đề cập?"
- Giảm Hallucination: AI có bằng chứng cứng từ đồ thị
"""

import os
import json
import pickle
import re
from typing import List, Dict, Any, Set, Optional

import networkx as nx

from src.core.database import db_instance
from src.core.logger import logger
from src.engine.generation.llm_client import LLMClient


class KnowledgeGraphBuilder:
    """Xây dựng Knowledge Graph từ các chunks đã Ingest.
    
    Pipeline:
    Chunks → LLM Extract Triples → NetworkX Graph → Pickle to Disk
    """

    GRAPH_DIR = os.path.join(os.getcwd(), "graph_store")

    def __init__(self):
        self.llm = LLMClient()
        os.makedirs(self.GRAPH_DIR, exist_ok=True)

    def _extract_triples(self, text: str, chunk_index: int) -> List[Dict]:
        """Dùng LLM trích xuất (Subject, Predicate, Object) từ 1 chunk."""
        prompt = f"""Trích xuất tất cả các mối quan hệ (triples) từ đoạn text sau.
Mỗi triple gồm: subject (chủ thể), predicate (quan hệ), object (đối tượng).

<text>
{text[:1500]}
</text>

Trả về JSON array, KHÔNG thêm markdown:
[{{"subject": "Eleven", "predicate": "có sức mạnh", "object": "telekinesis"}}, ...]

Quy tắc:
- Chỉ trích xuất thông tin CÓ TRONG TEXT, không bịa thêm.
- Subject và Object phải là danh từ riêng hoặc khái niệm cụ thể.
- Predicate phải là động từ hoặc cụm động từ ngắn gọn.
- Tối đa 8 triples cho mỗi đoạn.
- Nếu text quá ngắn hoặc không có thông tin, trả về []."""

        try:
            response = self.llm.chat_complete(
                prompt=prompt,
                system="Bạn là chuyên gia trích xuất tri thức. Chỉ trả về JSON array thuần, không markdown.",
                max_tokens=500,
                temperature=0.0
            )
            
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            triples = json.loads(response)
            
            # Gắn chunk_index vào mỗi triple
            for t in triples:
                t["chunk_index"] = chunk_index
            
            return triples

        except Exception as e:
            logger.warning(f"   ⚠️ Lỗi extract triples chunk {chunk_index}: {e}")
            return []

    def build_graph(self, collection_name: str) -> nx.DiGraph:
        """Xây dựng Knowledge Graph từ toàn bộ chunks trong Qdrant Collection."""
        logger.info(f"🕸️ [Graph RAG] Đang xây dựng Knowledge Graph cho [{collection_name}]...")
        
        # 1. Kéo chunks từ Qdrant
        records = []
        offset = None
        while True:
            result, next_offset = db_instance.client.scroll(
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

        if not records:
            logger.error("❌ Collection rỗng!")
            return nx.DiGraph()

        chunks = []
        for rec in records:
            payload = rec.payload or {}
            raw_text = payload.get("text", "")
            if "\n\n" in raw_text:
                raw_text = raw_text.split("\n\n")[-1]
            chunks.append({
                "id": rec.id,
                "content": raw_text,
                "index": payload.get("chunk_index", 0),
                "start_time": payload.get("start_time", 0.0),
            })
        chunks.sort(key=lambda x: x["index"])
        
        logger.info(f"   📦 Tổng chunks: {len(chunks)}")

        # 2. Trích mẫu để tiết kiệm API calls (Groq free tier)
        MAX_CHUNKS = 15
        if len(chunks) > MAX_CHUNKS:
            step = len(chunks) / MAX_CHUNKS
            sampled = [chunks[int(i * step)] for i in range(MAX_CHUNKS)]
        else:
            sampled = chunks
        
        logger.info(f"   🎯 Trích mẫu: {len(sampled)} chunks để extract triples")

        # 3. Extract triples bằng LLM
        G = nx.DiGraph()
        all_triples = []
        
        for idx, chunk in enumerate(sampled, 1):
            logger.info(f"   [{idx}/{len(sampled)}] Extracting triples từ chunk {chunk['index']}...")
            triples = self._extract_triples(chunk["content"], chunk["index"])
            all_triples.extend(triples)
            
            # Thêm vào Graph
            for t in triples:
                subj = t["subject"].strip()
                obj = t["object"].strip()
                pred = t["predicate"].strip()
                
                # Thêm Node với metadata
                if not G.has_node(subj):
                    G.add_node(subj, chunk_indices=set(), entity_type="entity")
                G.nodes[subj]["chunk_indices"].add(chunk["index"])
                
                if not G.has_node(obj):
                    G.add_node(obj, chunk_indices=set(), entity_type="entity")
                G.nodes[obj]["chunk_indices"].add(chunk["index"])
                
                # Thêm Edge (quan hệ)
                G.add_edge(subj, obj, relation=pred, chunk_index=chunk["index"])

        logger.info(f"   ✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        logger.info(f"   📊 Tổng triples: {len(all_triples)}")
        
        # 4. Lưu Graph xuống disk
        graph_path = os.path.join(self.GRAPH_DIR, f"{collection_name}.gpickle")
        
        # Convert sets to lists for pickling
        for node in G.nodes:
            if isinstance(G.nodes[node].get("chunk_indices"), set):
                G.nodes[node]["chunk_indices"] = list(G.nodes[node]["chunk_indices"])
        
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        
        # Lưu triples dạng JSON để debug
        triples_path = os.path.join(self.GRAPH_DIR, f"{collection_name}_triples.json")
        with open(triples_path, "w", encoding="utf-8") as f:
            json.dump(all_triples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"   💾 Đã lưu Graph tại: {graph_path}")
        return G


class GraphRetriever:
    """Truy xuất thông tin từ Knowledge Graph.
    
    Cách hoạt động:
    1. Nhận câu hỏi user → Trích Entity từ câu hỏi
    2. Tìm Node khớp trong Graph → Duyệt láng giềng (1-2 hop)
    3. Thu thập chunk_ids của các Node liên quan
    4. Kéo chunks từ Qdrant bằng chunk_ids
    """

    GRAPH_DIR = os.path.join(os.getcwd(), "graph_store")

    def __init__(self):
        self.llm = LLMClient()
        self.db = db_instance
        self._graph_cache: Dict[str, nx.DiGraph] = {}

    def _load_graph(self, collection_name: str) -> Optional[nx.DiGraph]:
        """Load Graph từ disk (có cache)."""
        if collection_name in self._graph_cache:
            return self._graph_cache[collection_name]
        
        graph_path = os.path.join(self.GRAPH_DIR, f"{collection_name}.gpickle")
        if not os.path.exists(graph_path):
            logger.warning(f"⚠️ Chưa có Knowledge Graph cho [{collection_name}]. Hãy chạy build trước.")
            return None
        
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        
        self._graph_cache[collection_name] = G
        logger.info(f"📂 Loaded Graph [{collection_name}]: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _extract_query_entities(self, query: str) -> List[str]:
        """Trích Entity từ câu hỏi user bằng LLM."""
        prompt = f"""Trích xuất các thực thể (entity) quan trọng từ câu hỏi sau:

Câu hỏi: "{query}"

Trả về JSON array các tên thực thể, KHÔNG markdown:
["Entity1", "Entity2", ...]

Nếu không có entity rõ ràng, trả về []."""

        try:
            response = self.llm.chat_complete(
                prompt=prompt,
                system="Trích xuất entity. Chỉ trả JSON array.",
                max_tokens=200,
                temperature=0.0
            )
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(response)
        except Exception:
            # Fallback: Tách từ viết hoa
            words = re.findall(r'\b[A-Z][a-zÀ-ỹ]+(?:\s+[A-Z][a-zÀ-ỹ]+)*\b', query)
            return words if words else query.split()[:3]

    def _find_matching_nodes(self, G: nx.DiGraph, entities: List[str]) -> Set[str]:
        """Tìm Node trong Graph khớp với Entity từ câu hỏi (Fuzzy matching)."""
        matched = set()
        graph_nodes = list(G.nodes)
        
        for entity in entities:
            entity_lower = entity.lower()
            for node in graph_nodes:
                node_lower = node.lower()
                # Exact match hoặc substring match
                if entity_lower in node_lower or node_lower in entity_lower:
                    matched.add(node)
        
        return matched

    def _get_subgraph_chunks(self, G: nx.DiGraph, seed_nodes: Set[str], max_hops: int = 2) -> Set[int]:
        """Duyệt Graph từ seed nodes, thu thập chunk_ids trong phạm vi N hops."""
        visited = set()
        chunk_indices = set()
        queue = [(node, 0) for node in seed_nodes]
        
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)
            
            # Thu thập chunk_indices từ node
            node_data = G.nodes.get(current, {})
            indices = node_data.get("chunk_indices", [])
            if isinstance(indices, (list, set)):
                chunk_indices.update(indices)
            
            # Duyệt láng giềng (cả in và out edges)
            if depth < max_hops:
                for neighbor in list(G.successors(current)) + list(G.predecessors(current)):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        return chunk_indices

    def _get_related_facts(self, G: nx.DiGraph, seed_nodes: Set[str]) -> List[str]:
        """Trích xuất các sự kiện (facts) từ edges liên quan đến seed nodes."""
        facts = []
        for node in seed_nodes:
            # Outgoing edges
            for _, target, data in G.out_edges(node, data=True):
                rel = data.get("relation", "liên quan đến")
                facts.append(f"{node} → {rel} → {target}")
            
            # Incoming edges
            for source, _, data in G.in_edges(node, data=True):
                rel = data.get("relation", "liên quan đến")
                facts.append(f"{source} → {rel} → {node}")
        
        return facts[:15]  # Giới hạn 15 facts

    def search(self, query: str, collection_name: str, top_k: int = 5) -> Dict[str, Any]:
        """Truy xuất Graph RAG: Entity → Graph Traversal + Global Stats.
        
        Returns:
            Dict chứa:
            - chunk_indices: Set[int] các chunk liên quan
            - facts: List[str] các sự kiện từ Graph
            - entities: List[str] entity được trích từ câu hỏi
            - matched_nodes: Set[str] nodes khớp trong Graph
            - graph_summary: str tóm tắt toàn bộ quy mô của đồ thị
        """
        logger.info(f"🕸️ Graph RAG Search: '{query}' → [{collection_name}]")
        
        G = self._load_graph(collection_name)
        if G is None:
            return {"chunk_indices": set(), "facts": [], "entities": [], "matched_nodes": set(), "graph_summary": ""}
        
        # 0. Lấy thống kê toàn cục để trị bệnh "mù chữ số" (Counting)
        # Chỉ lấy những node có tên riêng (viết hoa) hoặc có nhiều connections
        important_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:50]
        nodes_str = ", ".join([node for node, deg in important_nodes])
        
        graph_summary = f"""
THỐNG KÊ TRI THỨC TOÀN CỤC (GLOBAL GRAPH STATS):
- Tổng số thực thể (Entities): {G.number_of_nodes()}
- Tổng số quan hệ (Relationships): {G.number_of_edges()}
- Danh sách thực thể quan trọng nhất: {nodes_str}
"""
        
        # 1. Trích Entity từ câu hỏi
        entities = self._extract_query_entities(query)
        logger.info(f"   🔍 Entities từ query: {entities}")
        
        # 2. Tìm Node khớp
        matched = self._find_matching_nodes(G, entities)
        logger.info(f"   🎯 Matched nodes: {matched}")
        
        # 3. Duyệt Graph (2 hops)
        chunk_indices = set()
        facts = []
        if matched:
            chunk_indices = self._get_subgraph_chunks(G, matched, max_hops=2)
            facts = self._get_related_facts(G, matched)
        
        return {
            "chunk_indices": chunk_indices,
            "facts": facts,
            "entities": entities,
            "matched_nodes": matched,
            "graph_summary": graph_summary
        }

    def get_graph_summary(self, collection_name: str) -> str:
        """Tóm tắt Knowledge Graph (để inject vào Prompt LLM)."""
        G = self._load_graph(collection_name)
        if G is None:
            return ""
        
        # Top entities (nodes với nhiều connections nhất)
        degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
        top_entities = [f"{node} ({deg} connections)" for node, deg in degree_sorted]
        
        summary = f"Knowledge Graph: {G.number_of_nodes()} entities, {G.number_of_edges()} relationships.\n"
        summary += f"Top entities: {', '.join(top_entities)}"
        
        return summary
