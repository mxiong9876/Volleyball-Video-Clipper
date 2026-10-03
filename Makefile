.PHONY: install up down migrate dev test lint

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

# Postgres + Redis in Docker; API and web on the host with hot reload. Ctrl-C stops both.
dev: up
	@trap 'kill 0' INT TERM EXIT; \
	(cd api && uv run uvicorn app.main:app --reload --port 8000) & \
	(cd web && npm run dev) & \
	wait

test:
	cd pipeline && uv run pytest
	cd api && uv run pytest

lint:
	cd pipeline && uv run ruff check . && uv run ruff format --check .
	cd api && uv run ruff check . && uv run ruff format --check .
	cd web && npm run lint && npx tsc -b
