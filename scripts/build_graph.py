"""
Script xây dựng và test Knowledge Graph cho Graph RAG.

Sử dụng:
    # Bước 1: Build Knowledge Graph từ video đã Ingest
    poetry run python scripts/build_graph.py --build

    # Bước 2: Test truy vấn Graph RAG
    poetry run python scripts/build_graph.py --query "Eleven có sức mạnh gì?"

    # Hoặc chạy cả 2:
    poetry run python scripts/build_graph.py --build --query "Henry là ai?"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.database import db_instance
from src.core.logger import logger


def get_first_collection() -> str:
    cols = db_instance.client.get_collections().collections
    if not cols:
        logger.error("❌ Database Qdrant trống!")
        sys.exit(1)
    return cols[0].name


def run_build(collection_name: str):
    from src.engine.retrieval.graph_rag import KnowledgeGraphBuilder
    
    builder = KnowledgeGraphBuilder()
    G = builder.build_graph(collection_name)
    
    logger.info(f"\n🏆 KNOWLEDGE GRAPH ĐÃ HOÀN TẤT:")
    logger.info(f"   Nodes (Entities): {G.number_of_nodes()}")
    logger.info(f"   Edges (Relations): {G.number_of_edges()}")
    
    # In top entities
    degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"\n   🌟 Top 10 Entities Quan Trọng Nhất:")
    for i, (node, deg) in enumerate(degree_sorted, 1):
        logger.info(f"      [{i}] {node} ({deg} connections)")


def run_query(collection_name: str, query: str):
    from src.engine.retrieval.graph_rag import GraphRetriever
    
    retriever = GraphRetriever()
    result = retriever.search(query, collection_name)
    
    logger.info(f"\n🕸️ KẾT QUẢ GRAPH RAG SEARCH:")
    logger.info(f"   Query: {query}")
    logger.info(f"   Entities từ query: {result['entities']}")
    logger.info(f"   Matched Nodes: {result['matched_nodes']}")
    logger.info(f"   Chunk Indices: {result['chunk_indices']}")
    
    if result['facts']:
        logger.info(f"\n   📋 SỰ KIỆN TỪ ĐỒ THỊ:")
        for i, fact in enumerate(result['facts'], 1):
            logger.info(f"      [{i}] {fact}")
    else:
        logger.info(f"   ⚠️ Không tìm thấy sự kiện liên quan.")
    
    # Graph Summary
    summary = retriever.get_graph_summary(collection_name)
    logger.info(f"\n   📊 {summary}")


def main():
    parser = argparse.ArgumentParser(description="YouRAG Graph RAG Builder & Query")
    parser.add_argument("--build", action="store_true", help="Build Knowledge Graph")
    parser.add_argument("--query", type=str, default=None, help="Test query trên Graph")
    parser.add_argument("--collection", type=str, default=None, help="Collection name")
    
    args = parser.parse_args()
    collection = args.collection or get_first_collection()
    logger.info(f"📂 Collection: {collection}")
    
    if args.build:
        run_build(collection)
    
    if args.query:
        run_query(collection, args.query)
    
    if not args.build and not args.query:
        parser.print_help()
        print("\n💡 Ví dụ:")
        print('   poetry run python scripts/build_graph.py --build')
        print('   poetry run python scripts/build_graph.py --query "Eleven có sức mạnh gì?"')


if __name__ == "__main__":
    main()
