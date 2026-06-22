# syntax=docker/dockerfile:1.4
# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

# Install uv — Rust-based installer, 10-100x faster than pip/poetry
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN uv venv /app/.venv

# Install deps — BuildKit cache keeps packages between builds
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt --index-strategy unsafe-best-match

# ── Stage 2: Runner ───────────────────────────────────────────────────────────
FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    HF_HOME=/app/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

# Pre-download models into image layer.
# IMPORTANT: placed BEFORE "COPY src" so code changes don't invalidate this cache layer.
# Docker rebuilds này layer chỉ khi preload_models.py thay đổi (~2.5GB, 1 lần duy nhất).
COPY scripts/preload_models.py ./scripts/preload_models.py
RUN python scripts/preload_models.py

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
