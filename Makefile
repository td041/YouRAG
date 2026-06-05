DC = docker compose
export DOCKER_BUILDKIT=1

.PHONY: help setup build up down restart rebuild rebuild-fresh logs logs-backend status shell-backend shell-db test lint clean clean-all benchmark benchmark-generate benchmark-all

help:
	@echo "YouRAG Commands:"
	@echo "  make setup        — Tạo .env từ .env.example"
	@echo "  make build        — Build images"
	@echo "  make up           — Chạy hệ thống (background)"
	@echo "  make down         — Dừng hệ thống"
	@echo "  make rebuild       — Down + Build + Up (dùng pip cache, nhanh)"
	@echo "  make rebuild-fresh — Down + Build no-cache + Up (build từ đầu hoàn toàn)"
	@echo "  make logs         — Logs tất cả services"
	@echo "  make logs-backend — Logs backend"
	@echo "  make status       — Trạng thái containers + GPU"
	@echo "  make shell-backend— Vào container backend"
	@echo "  make shell-db     — Vào PostgreSQL"
	@echo "  make test              — Chạy tests"
	@echo "  make lint              — Kiểm tra code"
	@echo "  make benchmark         — Chạy RAGAS evaluation (collection đầu tiên)"
	@echo "  make benchmark COLLECTION=ten-collection — Chạy trên collection cụ thể"
	@echo "  make benchmark-generate— Sinh dataset 10 câu hỏi"
	@echo "  make benchmark-all     — Sinh dataset + evaluate luôn"
	@echo "  make clean        — Xóa containers"
	@echo "  make clean-all    — Xóa tất cả kể cả data ⚠️"
	@echo ""
	@echo "🧑‍💻 DEV MODE (Khuyên dùng):"
	@echo "  make dev-db       — Chỉ bật Database (Qdrant, Postgres, Redis) bằng Docker"
	@echo "  make dev-backend  — Chạy Backend ở máy thật (có tự động reload code)"
	@echo "  make dev-frontend — Chạy Frontend ở máy thật"

setup:
	@[ -f .env ] && echo ".env đã tồn tại." || (cp .env.example .env && echo "✅ Đã tạo .env — điền GROQ_API_KEY vào!")

build:
	$(DC) build

up:
	$(DC) up -d
	@echo "✅ Backend: http://localhost:8000 | Frontend: http://localhost:3000 | Docs: http://localhost:8000/docs"

# ─── MÔI TRƯỜNG LẬP TRÌNH (DEV MODE) ──────────────────────────────
dev-db:
	$(DC) up -d postgres qdrant redis
	@echo "✅ Đã bật Database nội bộ. Chạy 'make dev-backend' ở tab mới nhé!"

dev-backend:
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev
# ──────────────────────────────────────────────────────────────────

down:
	$(DC) down

restart:
	$(DC) restart

rebuild:
	$(DC) down
	$(DC) build --pull=false
	$(DC) up -d

# Dùng khi thay đổi Dockerfile hoặc pyproject.toml (không dùng thường xuyên)
rebuild-fresh:
	$(DC) down
	$(DC) build --no-cache --pull
	$(DC) up -d

logs:
	$(DC) logs -f

logs-backend:
	$(DC) logs -f backend

status:
	$(DC) ps
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU không khả dụng"

shell-backend:
	$(DC) exec backend bash

shell-db:
	$(DC) exec postgres psql -U yourag_user -d yourag_db

test:
	poetry run pytest tests/ -v --cov=src

lint:
	poetry run ruff check src/

# ─── BENCHMARK ────────────────────────────────────────────────────────
COLLECTION ?= $(shell python3 -c "import requests; cols=requests.get('http://localhost:8000/collections').json(); print(cols[0]['name'] if cols else '')" 2>/dev/null)

benchmark:
	@echo "📊 Chạy RAGAS benchmark trên collection: $(COLLECTION)"
	poetry run python tests/run_benchmark.py --evaluate --collection $(COLLECTION)

benchmark-generate:
	@echo "📝 Sinh dataset câu hỏi (10 câu) cho collection: $(COLLECTION)"
	poetry run python tests/run_benchmark.py --generate --collection $(COLLECTION) --num-questions 10

benchmark-all:
	@echo "🚀 Sinh dataset + chạy RAGAS benchmark cho collection: $(COLLECTION)"
	poetry run python tests/run_benchmark.py --all --collection $(COLLECTION) --num-questions 10

clean:
	$(DC) down --remove-orphans

clean-all:
	@read -p "Xóa toàn bộ data? (yes/no): " c && [ "$$c" = "yes" ] && $(DC) down -v --remove-orphans || echo "Hủy."
