# syntax=docker/dockerfile:1.4
# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

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
    VIRTUAL_ENV="/app/.venv"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
