"""Rally segmentation from whistle times (baseline: audio only).

A rally runs from one whistle (serve) to the next (point over). Whistles closer than
`dedupe` s are one blow; pairing is greedy, and a gap longer than `max_rally` means the open
whistle was a stray (timeout, sub, missed end whistle) and the next whistle starts over.
Run the detector on every video in data/raw with:

    uv run python -m volley_pipeline.segment --raw ../data/raw --out ../data/predictions \
        --run <run_name>
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from volley_pipeline import audio, videos
from volley_pipeline.eval import Rally
from volley_pipeline.ingest import AUDIO_FILENAME

MIN_RALLY_SEC = 1.0
MAX_RALLY_SEC = 45.0
DEDUPE_SEC = 1.0


def dedupe_whistles(times: Sequence[float], window: float = DEDUPE_SEC) -> list[float]:
    """Sorted whistle times with any time within `window` s of the previous kept one dropped."""
    kept: list[float] = []
    for t in sorted(times):
        if not kept or t - kept[-1] > window:
            kept.append(t)
    return kept


def whistles_to_rallies(
    whistle_times: Sequence[float],
    *,
    min_rally: float = MIN_RALLY_SEC,
    max_rally: float = MAX_RALLY_SEC,
    dedupe: float = DEDUPE_SEC,
) -> list[Rally]:
    """Pair consecutive whistles into (start, end) rallies."""
    rallies: list[Rally] = []
    open_t: float | None = None
    for t in dedupe_whistles(whistle_times, dedupe):
        if open_t is None:
            open_t = t
        elif min_rally <= t - open_t <= max_rally:
            rallies.append((open_t, t))
            open_t = None
        elif t - open_t > max_rally:
            open_t = t  # previous whistle had no partner; this one may be a serve
        # gap < min_rally can't happen after dedupe unless min_rally > dedupe; ignore it
    return rallies


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_video(audio_path: Path, out_dir: Path, youtube_id: str) -> tuple[int, int]:
    """Detect, segment and write <id>.json + <id>.whistles.json; returns (whistles, rallies)."""
    whistles = audio.detect_whistles_in_file(audio_path)
    rallies = whistles_to_rallies([w.start_sec for w in whistles])
    _write_json(
        out_dir / f"{youtube_id}.json",
        [{"start_sec": round(s, 2), "end_sec": round(e, 2)} for s, e in rallies],
    )
    _write_json(out_dir / f"{youtube_id}.whistles.json", [asdict(w) for w in whistles])
    return len(whistles), len(rallies)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Whistle-based rally detection for data/raw.")
    parser.add_argument("--raw", default="data/raw", help="dir of <youtube_id>/audio.wav")
    parser.add_argument("--out", default="data/predictions", help="predictions root dir")
    parser.add_argument("--run", required=True, help="run name; writes <out>/<run>/<id>.json")
    parser.add_argument("--videos-doc", help="videos table (default: docs/videos.md)")
    parser.add_argument("--include-held-out", action="store_true", help="also run held-out")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw)
    out_dir = Path(args.out) / args.run
    try:
        held_out = videos.load_held_out(args.videos_doc, raw_dir, args.include_held_out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    wavs = sorted(raw_dir.glob(f"*/{AUDIO_FILENAME}"))
    if not wavs:
        print(f"error: no */{AUDIO_FILENAME} in {raw_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "_run.json",
        {
            "detector": "whistle",
            "band_hz": list(audio.BAND_HZ),
            "frame": audio.FRAME,
            "hop": audio.HOP,
            "min_rally_sec": MIN_RALLY_SEC,
            "max_rally_sec": MAX_RALLY_SEC,
            "dedupe_sec": DEDUPE_SEC,
        },
    )
    for wav in wavs:
        vid = wav.parent.name
        if vid in held_out:
            print(f"{vid}: held out, skipped")
            continue
        n_whistles, n_rallies = run_video(wav, out_dir, vid)
        print(f"{vid}: {n_whistles} whistles -> {n_rallies} rallies")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
