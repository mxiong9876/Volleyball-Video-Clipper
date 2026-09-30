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

## Commands
(Claude: keep this section updated as commands are created)

## Rules
- Only work on the phase I ask for. Don't start later phases.
- pipeline/ functions take file paths or arrays and return plain data. Unit-test them.
- Never commit video/audio files or .env. Test media lives in data/raw/.
- Any change to rally detection must be scored with the eval script against
  data/labels/ and report precision/recall before and after.
- Ask before adding a dependency not listed above.
- Commit after each working step with a clear message.
