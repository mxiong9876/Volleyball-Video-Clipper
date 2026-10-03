---
name: project-map
description: Compact map of the Volley Breakdown repo (what each file/folder does, how api/pipeline/web connect, main make commands). Use before searching the codebase, when unsure where something lives, or when starting work on a new phase.
---

# Project map

## Root
- `CLAUDE.md` - project rules, stack, commands
- `docs/BLUEPRINT.md` - full product spec and development phases
- `Makefile` - install/up/down/dev/test/lint
- `docker-compose.yml` - Postgres 16 + Redis 7 with healthchecks
- `pyproject.toml` - uv workspace root (virtual; members: api, pipeline)
- `uv.lock` - single lockfile for the whole Python workspace
- `.env.example` - DATABASE_URL, REDIS_URL for the api (copy to `.env`)
- `data/labels/` - committed rally labels for eval; `data/raw/` holds local test media (gitignored)

## Python workspace
api/ and pipeline/ are one uv workspace: root `pyproject.toml`, one root `.venv`
and `uv.lock`. The pipeline package is `volley_pipeline` (dist name
`volley-pipeline`); api depends on it via `[tool.uv.sources] workspace = true`.

## api/ (FastAPI, not packaged)
- `api/pyproject.toml` - deps, dev group (pytest, ruff, httpx2), ruff/pytest config
- `api/app/main.py` - FastAPI app, CORS, `/health` (checks Postgres + Redis, 503 if down)
- `api/app/db.py` - SQLAlchemy `Base`, cached `get_engine()`, `SessionLocal()`, `get_session` dependency
- `api/app/models.py` - `Video` (unique youtube_id), `AnalysisJob` (table analysis_jobs),
  `JobStatus` StrEnum (VARCHAR + CHECK constraint, not a PG enum), `TERMINAL_STATUSES`
- `api/alembic.ini`, `api/migrations/env.py` - Alembic; URL from app.config unless set explicitly
- `api/migrations/versions/` - migrations (`0001` = videos + analysis_jobs)
- `api/app/config.py` - env-driven settings (DATABASE_URL, REDIS_URL, CORS_ORIGINS)
- `api/tests/test_health.py` - /health ok + degraded cases (checks monkeypatched)
- `api/tests/test_pipeline_import.py` - proves api can import volley_pipeline
- `api/tests/test_migrations.py` - upgrade/downgrade on Postgres DB `volley_test` (skips if down)

## pipeline/ (pure-Python analysis, no web imports)
- `pipeline/pyproject.toml` - hatchling package `volley_pipeline`
- `pipeline/volley_pipeline/__init__.py` - package root, `__version__`
- `pipeline/tests/test_smoke.py` - import smoke test

## web/ (React + Vite + TS + Tailwind + TanStack Query)
- `web/src/main.tsx` - React root, QueryClientProvider
- `web/src/App.tsx` - health status page (polls /health every 5s)
- `web/src/api.ts` - API client; VITE_API_URL, falls back to http://localhost:8000
- `web/src/index.css` - Tailwind entry
- `web/vite.config.ts` - Vite + React + Tailwind plugins
- `web/.env.example` - VITE_API_URL (Vite reads env only from web/)
- `web/.oxlintrc.json` - oxlint config

## Data flow (planned, per BLUEPRINT; only /health exists today)
User pastes a YouTube URL in web → api creates a job and enqueues it on Redis (RQ)
→ worker runs volley_pipeline stages (download, rallies, OCR) → results saved to
Postgres → web fetches them for the momentum chart and rally playback.

## Commands
- `make install` - `uv sync --all-packages` + `npm install` in web/
- `make up` / `make down` - start/stop Postgres + Redis
- `make migrate` - alembic upgrade head on the dev DB
- `make dev` - up, then API :8000 (reload) + Vite :5173
- `make test` - pytest in pipeline/ and api/
- `make lint` - ruff check/format (Python), oxlint + tsc (web)
