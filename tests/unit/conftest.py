"""
conftest.py — patches QdrantClient and SentenceTransformer at MODULE LOAD TIME
so they are intercepted before pytest collects any test file.

Why here (module level, not inside a fixture):
  src.core.database runs `db_instance = VectorDatabase()` at import time,
  which opens QdrantClient + loads BAAI/bge-m3 (1.5 GB).
  pytest collects test files (imports them) BEFORE any fixture runs,
  so module-level patches are the only way to prevent the real I/O.
"""
from unittest.mock import MagicMock, patch

# ── 1. Patch SentenceTransformer (blocks BAAI/bge-m3 download) ────────────────
_st_patcher = patch("sentence_transformers.SentenceTransformer", autospec=False)
MockSentenceTransformer = _st_patcher.start()
MockSentenceTransformer.return_value = MagicMock()

# ── 2. Patch QdrantClient (blocks local qdrant_db lock) ───────────────────────
_qdrant_patcher = patch("qdrant_client.QdrantClient", autospec=False)
MockQdrantClient = _qdrant_patcher.start()
MockQdrantClient.return_value = MagicMock()

# ── 3. Patch CrossEncoder (blocks mmarco model download) ──────────────────────
_ce_patcher = patch("sentence_transformers.CrossEncoder", autospec=False)
MockCrossEncoder = _ce_patcher.start()
MockCrossEncoder.return_value = MagicMock()
