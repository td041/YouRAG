# YouRAG — Deployment Guide

Deploy YouRAG to production: **Railway** (backend) + **Vercel** (frontend) + **Qdrant Cloud** (vector DB).

---

## Prerequisites

- [Railway](https://railway.app) account
- [Vercel](https://vercel.com) account
- [Qdrant Cloud](https://cloud.qdrant.io) account (free 1GB cluster)
- [Groq API Key](https://console.groq.com/keys)
- [Mistral API Key](https://console.mistral.ai) (for RAGAS benchmark)

---

## Step 1 — Qdrant Cloud

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io) → **Create Cluster** → Free tier
2. Copy **Cluster URL** (`https://xxxx.aws.cloud.qdrant.io`) and **API Key**
3. Save for Step 2

---

## Step 2 — Railway (Backend)

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
2. Select `YouRAG` repo — Railway auto-detects `railway.toml`
3. Add services:
   - **Redis** → Add Plugin → Redis
   - **PostgreSQL** → Add Plugin → PostgreSQL
4. Set environment variables:

```
GROQ_API_KEY=gsk_...
MISTRAL_EVAL_API_KEY=...
QDRANT_SERVER_URL=https://xxxx.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
API_KEY=generate_a_strong_random_string
LLM_PROVIDER=groq
DEVICE=cpu
ENVIRONMENT=production
```

> Generate `API_KEY`: `python -c "import secrets; print(secrets.token_hex(32))"`

5. Deploy — Railway uses `railway.toml` → `/health` endpoint for health checks

---

## Step 3 — Vercel (Frontend)

1. Go to [vercel.com](https://vercel.com) → **New Project** → Import `YouRAG`
2. Set **Root Directory**: `frontend`
3. Set environment variables:

```
NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app
NEXT_PUBLIC_API_KEY=same_value_as_API_KEY_above
```

4. Deploy

---

## Step 4 — Verify

```bash
# Health check
curl https://your-backend.up.railway.app/health

# Test ingest
curl -X POST https://your-backend.up.railway.app/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

---

## Local Development

```bash
cp .env.example .env
# Fill in GROQ_API_KEY at minimum

# Option A: Full Docker stack
docker compose up -d --build

# Option B: Dev mode (hot reload)
docker compose up -d postgres qdrant redis   # databases only
poetry run uvicorn src.api.main:app --reload --port 8000
cd frontend && npm run dev
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Primary LLM (llama-3.3-70b) |
| `GROQ_API_KEYS` | Optional | Backup keys for benchmark rotation (comma-separated) |
| `API_KEY` | Production | Protects all write endpoints |
| `NEXT_PUBLIC_API_KEY` | Production | Frontend uses same value |
| `MISTRAL_EVAL_API_KEY` | Benchmark | RAGAS evaluator (no daily quota) |
| `JINA_API_KEY` | Optional | Late Chunking embeddings (1M tokens free) |
| `QDRANT_SERVER_URL` | Production | Qdrant Cloud URL |
| `QDRANT_API_KEY` | Production | Qdrant Cloud API key |
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `REDIS_URL` | ✅ | Redis connection string |
| `DEVICE` | Optional | `cpu` or `cuda` (default: `cpu` for cloud) |
| `ENVIRONMENT` | Optional | `development` / `production` |

---

## Docker Compose (Local)

```yaml
# 5 services
services:
  backend   # FastAPI + AI models (port 8000)
  frontend  # Next.js UI (port 3000)
  qdrant    # Vector DB (port 6333)
  postgres  # Chat history (port 5432)
  redis     # Cache + job store (port 6379)
```

All data persisted in named Docker volumes — survives container restarts.
