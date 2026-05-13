DC = docker compose
export DOCKER_BUILDKIT=1

.PHONY: help setup build up down restart rebuild rebuild-fresh logs logs-backend status shell-backend shell-db test lint clean clean-all

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
	@echo "  make test         — Chạy tests"
	@echo "  make lint         — Kiểm tra code"
	@echo "  make clean        — Xóa containers"
	@echo "  make clean-all    — Xóa tất cả kể cả data ⚠️"

setup:
	@[ -f .env ] && echo ".env đã tồn tại." || (cp .env.example .env && echo "✅ Đã tạo .env — điền GROQ_API_KEY vào!")

build:
	$(DC) build

up:
	$(DC) up -d
	@echo "✅ Backend: http://localhost:8000 | Frontend: http://localhost:3000 | Docs: http://localhost:8000/docs"

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

clean:
	$(DC) down --remove-orphans

clean-all:
	@read -p "Xóa toàn bộ data? (yes/no): " c && [ "$$c" = "yes" ] && $(DC) down -v --remove-orphans || echo "Hủy."
