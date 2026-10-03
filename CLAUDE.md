# Volley Breakdown

## What this is
Web app: paste a YouTube volleyball match URL → detect rallies, OCR the
scoreboard, show a momentum chart, play rallies via the YouTube embed.
Full spec: docs/BLUEPRINT.md. We build one phase at a time.

## Structure
- web/      React + Vite + TypeScript + Tailwind, TanStack Query, Recharts
- api/      FastAPI (Python 3.11), SQLAlchemy, Postgres, Redis + RQ worker
- pipeline/ Pure-Python analysis (yt-dlp, ffmpeg, PySceneDetect, librosa,
            OpenCV, PaddleOCR). No web imports. Every stage runnable from CLI.

## Stack (approved tooling, in addition to the above)
- uv workspace: root pyproject.toml lists api/ + pipeline/; one uv.lock and .venv at
  the root. api depends on volley-pipeline (workspace source). Python 3.11
- api runtime: uvicorn, psycopg[binary] (Postgres driver), redis, alembic, rq
- Python dev group (`[dependency-groups] dev`): pytest, ruff, httpx2 (api only;
  Starlette's TestClient deprecated plain httpx)
- web lint: oxlint (Vite template default)

## Commands
(Claude: keep this section updated as commands are created)
- Prereq: ffmpeg on PATH (`brew install ffmpeg`); used by yt-dlp and the audio stage/tests
- `make install` - `uv sync --all-packages` (whole workspace), npm install web/
- `make up` / `make down` - start/stop Postgres + Redis (docker compose)
- `make migrate` - `alembic upgrade head` on the dev DB (run after `make up`, on schema changes)
- New migration: `cd api && uv run alembic revision --autogenerate -m "..."`
  (models in api/app/models.py; review the generated file before committing)
- `make dev` - `up`, then API on :8000 (reload) + Vite on :5173; Ctrl-C stops both
- `make test` - pytest in pipeline/ and api/ (api/tests/test_migrations.py needs
  `make up`; it uses a separate `volley_test` DB and skips if Postgres is down)
- `make lint` - ruff check + format check (Python), oxlint + tsc (web)
- Ingest CLI: `cd pipeline && uv run python -m volley_pipeline.ingest <url> --out ../data/raw`
  (writes data/raw/<youtube_id>/video.mp4 + audio.wav)
- Single test: `cd api && uv run pytest tests/test_health.py::test_health_ok`
- Env: copy `.env.example` → `.env` (api) and `web/.env.example` → `web/.env`
  (Vite only reads web/; VITE_API_URL defaults to http://localhost:8000)
- Health: `curl -i localhost:8000/health` (200 ok, 503 if db or redis down)

## Rules
- Only work on the phase I ask for. Don't start later phases.
- pipeline/ functions take file paths or arrays and return plain data. Unit-test them.
- Never commit video/audio files or .env. Test media lives in data/raw/.
- Any change to rally detection must be scored with the eval script against
  data/labels/ and report precision/recall before and after.
- Ask before adding a dependency not listed above.
- Commit after each working step with a clear message.
- When you add, move, or delete a file in api/, pipeline/, or web/src/, update
  .claude/skills/project-map/SKILL.md in the same commit.
- Never modify or delete an existing test to make it pass. If a test
  seems wrong, stop and tell me why.
