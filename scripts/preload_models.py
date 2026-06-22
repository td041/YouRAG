"""Pre-download all ML models into the Docker image at build time.

Chạy trong Dockerfile RUN step — models được bake vào image layer,
tránh tải lại mỗi lần container restart.

Models:
  1. BAAI/bge-m3             — embedding (~1.1GB)
  2. BAAI/bge-reranker-v2-m3 — cross-encoder reranker (~570MB)
  3. naver/efficient-splade-VI-BT-large-doc   — SPLADE sparse (~440MB)
  4. naver/efficient-splade-VI-BT-large-query — SPLADE sparse (~440MB)

Total: ~2.5GB
"""
import os
import sys

os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/app/models")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 60, flush=True)
print(f"🔽 YouRAG — Pre-downloading models to {os.environ['HF_HOME']}", flush=True)
print("=" * 60, flush=True)

# 1. Embedding model — used by VectorDatabase (src/core/database.py)
print("\n[1/4] BAAI/bge-m3  (~1.1 GB)...", flush=True)
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-m3")
print("  ✅ bge-m3 ready", flush=True)

# 2. Cross-encoder reranker — used by CrossEncoderReranker (src/engine/ranking/cross_encoder.py)
print("\n[2/4] BAAI/bge-reranker-v2-m3  (~570 MB)...", flush=True)
from sentence_transformers import CrossEncoder
CrossEncoder("BAAI/bge-reranker-v2-m3")
print("  ✅ bge-reranker-v2-m3 ready", flush=True)

# 3. SPLADE doc encoder — used by SpladeRetriever (src/engine/retrieval/splade_search.py)
print("\n[3/4] naver/efficient-splade-VI-BT-large-doc  (~440 MB)...", flush=True)
from transformers import AutoModelForMaskedLM, AutoTokenizer
AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-doc")
AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-doc", low_cpu_mem_usage=False)
print("  ✅ SPLADE doc encoder ready", flush=True)

# 4. SPLADE query encoder
print("\n[4/4] naver/efficient-splade-VI-BT-large-query  (~440 MB)...", flush=True)
AutoTokenizer.from_pretrained("naver/efficient-splade-VI-BT-large-query")
AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-VI-BT-large-query", low_cpu_mem_usage=False)
print("  ✅ SPLADE query encoder ready", flush=True)

print(f"\n✅ All models ready → {os.environ['HF_HOME']}", flush=True)
