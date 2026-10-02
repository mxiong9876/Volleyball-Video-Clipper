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
- uv: Python 3.11 + venvs (api/ and pipeline/ each have pyproject.toml + uv.lock)
- api runtime: uvicorn, psycopg[binary] (Postgres driver), redis
- Python dev group (`[dependency-groups] dev`): pytest, ruff, httpx
- web lint: oxlint (Vite template default)

## Commands
(Claude: keep this section updated as commands are created)
- `make install` - uv sync api/ + pipeline/, npm install web/
- `make up` / `make down` - start/stop Postgres + Redis (docker compose)
- `make dev` - `up`, then API on :8000 (reload) + Vite on :5173; Ctrl-C stops both
- `make test` - pytest in pipeline/ and api/
- `make lint` - ruff check + format check (Python), oxlint + tsc (web)
- Single test: `cd api && uv run pytest tests/test_health.py::test_health_ok`
- Health: `curl -i localhost:8000/health` (200 ok, 503 if db or redis down)

## Rules
- Only work on the phase I ask for. Don't start later phases.
- pipeline/ functions take file paths or arrays and return plain data. Unit-test them.
- Never commit video/audio files or .env. Test media lives in data/raw/.
- Any change to rally detection must be scored with the eval script against
  data/labels/ and report precision/recall before and after.
- Ask before adding a dependency not listed above.
- Commit after each working step with a clear message.
- Never modify or delete an existing test to make it pass. If a test
  seems wrong, stop and tell me why.
