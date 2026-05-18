# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# Cài đặt dependencies vào venv riêng; build-essential chỉ cần ở đây (psycopg2, etc.)
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && rm -rf /var/lib/apt/lists/*

# Copy lock files trước để tận dụng Docker layer cache
# Layer này chỉ rebuild khi pyproject.toml / poetry.lock thay đổi
COPY pyproject.toml poetry.lock ./

# Cài runtime deps: bỏ dev + evaluation (ZenML/RAGAS ~500MB không cần cho server)
RUN poetry install --no-root --without dev,evaluation --no-cache

# ── Stage 2: Runner ───────────────────────────────────────────────────────────
# Image cuối không có Poetry, build-essential, hay pip — nhỏ hơn ~400MB
FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

WORKDIR /app

# libpq cho psycopg2, curl cho Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv đã build sẵn từ stage builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code (được lọc bởi .dockerignore)
COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
