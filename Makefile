.PHONY: install up down migrate worker dev test lint

install:
	uv sync --all-packages
	cd web && npm install

up:
	docker compose up -d --wait

down:
	docker compose down

# Apply Alembic migrations to the dev database.
migrate: up
	cd api && uv run alembic upgrade head

# RQ worker for ingest jobs (fails jobs a previous worker left unfinished on startup).
worker: up
	cd api && uv run python worker.py

# Postgres + Redis in Docker; API, worker and web on the host. Ctrl-C stops all three.
dev: up
	@trap 'kill 0' INT TERM EXIT; \
	(cd api && uv run uvicorn app.main:app --reload --port 8000) & \
	(cd api && uv run python worker.py) & \
	(cd web && npm run dev) & \
	wait

test:
	cd pipeline && uv run pytest
	cd api && uv run pytest

lint:
	cd pipeline && uv run ruff check . && uv run ruff format --check .
	cd api && uv run ruff check . && uv run ruff format --check .
	cd web && npm run lint && npx tsc -b
