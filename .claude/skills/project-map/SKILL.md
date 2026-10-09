---
name: project-map
description: Compact map of the Volley Breakdown repo (what each file/folder does, how api/pipeline/web connect, main make commands). Use before searching the codebase, when unsure where something lives, or when starting work on a new phase.
---

# Project map

## Root
- `CLAUDE.md` - project rules, stack, commands
- `docs/BLUEPRINT.md` - full product spec and development phases
- `docs/videos.md` - test video table (ID, angle, whistle, Split tune/held-out, label status)
- `docs/labeling-rules.md` - rally start/end definitions used by the labels
- `Makefile` - install/up/down/dev/test/lint
- `docker-compose.yml` - Postgres 16 + Redis 7 with healthchecks
- `pyproject.toml` - uv workspace root (virtual; members: api, pipeline)
- `uv.lock` - single lockfile for the whole Python workspace
- `.env.example` - DATABASE_URL, REDIS_URL for the api (copy to `.env`)
- `data/labels/` - committed rally labels for eval; `data/raw/` holds local test media and
  `data/predictions/<run>/` detector output (both gitignored)
- `tools/labeler/index.html` - standalone Chrome page (File System Access API) for hand-labeling
  rallies (S/E/Z keys); reads data/raw/<id>/video.mp4, autosaves data/labels/<id>.json
  (`[{start_sec, end_sec}]`). Usage in `tools/labeler/README.md`

## Python workspace
api/ and pipeline/ are one uv workspace: root `pyproject.toml`, one root `.venv`
and `uv.lock`. The pipeline package is `volley_pipeline` (dist name
`volley-pipeline`); api depends on it via `[tool.uv.sources] workspace = true`.

## api/ (FastAPI, not packaged)
- `api/pyproject.toml` - deps, dev group (pytest, ruff, httpx2), ruff/pytest config
- `api/app/main.py` - FastAPI app, CORS, `/health` (checks Postgres + Redis, 503 if down)
- `api/app/db.py` - SQLAlchemy `Base`, cached `get_engine()`, `SessionLocal()`, `get_session`/`SessionDep`
- `api/app/models.py` - `Video` (unique youtube_id), `AnalysisJob` (table analysis_jobs),
  `JobStatus` StrEnum (VARCHAR + CHECK constraint, not a PG enum), `TERMINAL_STATUSES`,
  `UTCDateTime` (timestamps always read back tz-aware)
- `api/alembic.ini`, `api/migrations/env.py` - Alembic; URL from app.config unless set explicitly
- `api/migrations/versions/` - migrations (`0001` = videos + analysis_jobs)
- `api/app/config.py` - env-driven settings (DATABASE_URL, REDIS_URL, CORS_ORIGINS, DATA_DIR)
  plus limits: MAX_DURATION_SEC (3h), METADATA_TIMEOUT, QUEUE_NAME, INGEST_JOB_TIMEOUT
- `api/app/queue.py` - Redis/RQ queue, `enqueue_ingest` (RQ id `ingest-<job_id>`),
  worker heartbeat key (30s TTL), `rq_job_is_live` (waiting, or started on a worker whose
  heartbeat is fresh; RQ's own worker registry outlives a kill -9 by hours)
- `api/app/tasks.py` - `ingest_video(job_id)` (calls volley_pipeline.ingest, updates status/
  progress), `fail_stale_jobs`, `mark_failed`
- `api/app/routes/videos.py` - POST /videos: parse id -> reuse/retry existing -> lookup (deadline)
  -> 3h check -> insert + enqueue. 202 new, 200 existing, 422 bad/unavailable/too long, 503, 504
- `api/app/routes/jobs.py` - GET /jobs/{id} (`JobOut`), 404 if missing
- `api/worker.py` - RQ SimpleWorker entry point; runs `fail_stale_jobs` first, heartbeat thread
- `api/tests/test_health.py` - /health ok + degraded cases (checks monkeypatched)
- `api/tests/test_pipeline_import.py` - proves api can import volley_pipeline
- `api/tests/test_migrations.py` - upgrade/downgrade on Postgres DB `volley_test` (skips if down)
- `api/tests/conftest.py` - in-memory SQLite fixtures (`session`, `client`), `add_video` helper
- `api/tests/test_videos.py`, `test_jobs.py`, `test_tasks.py`, `test_queue.py` - endpoints,
  task status flow + stale-job recovery, RQ liveness (lookup/enqueue/RQ all faked)

## pipeline/ (pure-Python analysis, no web imports)
- `pipeline/pyproject.toml` - hatchling package `volley_pipeline` (deps: yt-dlp, numpy, scipy)
- `pipeline/volley_pipeline/__init__.py` - package root, `__version__`
- `pipeline/volley_pipeline/ingest.py` - `parse_youtube_id` (no network), `fetch_metadata`,
  `check_duration` (3h cap), `download_video` (-> video.mp4, cleans partials),
  `extract_audio` (ffmpeg -> mono 22050 Hz WAV), `IngestError` subclasses; CLI `python -m`
- `pipeline/volley_pipeline/eval.py` - rally eval: `load_rallies` (labeler JSON), `match_rallies`
  (max one-to-one match, both boundaries within tolerance, default 1.5s), `score_video` -> `Scores`
  (tp/fp/fn, P/R/F1, mean signed/abs start+end offsets), `evaluate` (per video + micro overall); CLI
- `pipeline/volley_pipeline/audio.py` - whistle detection: `load_wav` (mono float32),
  `whistle_activity` (chunked STFT: in-band 2-4 kHz peak-vs-median dB, band energy share),
  `detect_whistles` -> `Whistle(start_sec, end_sec, peak_hz, strength_db)`, `detect_whistles_in_file`
- `pipeline/volley_pipeline/segment.py` - `whistles_to_rallies` (dedupe, greedy pairing of
  consecutive whistles); CLI runs every non-held-out data/raw/<id>/audio.wav and writes
  data/predictions/<run>/<id>.json (labels format), <id>.whistles.json, _run.json
- `pipeline/volley_pipeline/videos.py` - `held_out_ids` (parses the Split column of
  docs/videos.md), `default_videos_doc` (data/<x> -> docs/videos.md), `load_held_out` (CLI helper)
- `pipeline/tests/test_smoke.py` - import smoke test
- `pipeline/tests/test_eval.py` - eval tests on hand-made rally lists (tolerance edges, offsets, CLI)
- `pipeline/tests/test_audio.py` - whistle detection on synthetic tones/trills/noise (no real media)
- `pipeline/tests/test_segment.py` - whistle pairing rules + CLI on generated WAVs (held-out skip)
- `pipeline/tests/test_videos.py` - videos.md table parsing (held-out ids, malformed rows)
- `pipeline/tests/test_ingest.py` - ingest tests; yt-dlp faked, ffmpeg run on a generated 1s clip

## web/ (React + Vite + TS + Tailwind + TanStack Query)
- `web/src/main.tsx` - React root, QueryClientProvider
- `web/src/App.tsx` - health status page (polls /health every 5s)
- `web/src/api.ts` - API client; VITE_API_URL, falls back to http://localhost:8000
- `web/src/index.css` - Tailwind entry
- `web/vite.config.ts` - Vite + React + Tailwind plugins
- `web/.env.example` - VITE_API_URL (Vite reads env only from web/)
- `web/.oxlintrc.json` - oxlint config

## Data flow
Built (Phase 1): POST /videos → api validates + looks up metadata → inserts videos +
analysis_jobs row → enqueues on Redis (RQ) → worker runs volley_pipeline.ingest (download
→ data/raw/<youtube_id>/video.mp4, audio.wav) and updates status/progress → GET /jobs/{id}.
Phase 2 (in progress): audio.wav → audio.detect_whistles → segment.whistles_to_rallies →
data/predictions/<run>/ → eval / diagnose against data/labels.
Planned: OCR, results in Postgres → web momentum chart + playback.

## Commands
- `make install` - `uv sync --all-packages` + `npm install` in web/
- `make up` / `make down` - start/stop Postgres + Redis
- `make migrate` - alembic upgrade head on the dev DB
- `make worker` - RQ ingest worker (also started by `make dev`)
- `make dev` - up, then API :8000 (reload) + worker + Vite :5173
- `make test` - pytest in pipeline/ and api/
- `make lint` - ruff check/format (Python), oxlint + tsc (web)
