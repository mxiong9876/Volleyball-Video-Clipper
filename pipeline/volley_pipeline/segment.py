"""Rally segmentation: whistle (audio) or motion (video) detector over data/raw.

Whistle: a rally runs from one whistle (serve) to the next (point over). Whistles closer than
`dedupe` s are one blow; pairing is greedy, and a gap longer than `max_rally` means the open
whistle was a stray (timeout, sub, missed end whistle) and the next whistle starts over.
Motion: active stretches of frame-difference energy with camera cuts and occlusions (people
walking in front of the camera: one large connected changed region) masked out (see
volley_pipeline.motion). Run a detector on every video in data/raw with:

    uv run python -m volley_pipeline.segment --raw ../data/raw --out ../data/predictions \
        --run <run_name> [--detector motion]
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from volley_pipeline import audio, motion, videos
from volley_pipeline.eval import Rally
from volley_pipeline.ingest import AUDIO_FILENAME, VIDEO_FILENAME

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


def _write_rallies(path: Path, rallies: Sequence[Rally]) -> None:
    _write_json(path, [{"start_sec": round(s, 2), "end_sec": round(e, 2)} for s, e in rallies])


def run_whistle_video(audio_path: Path, out_dir: Path, youtube_id: str) -> str:
    """Detect, segment and write <id>.json + <id>.whistles.json; returns a summary line."""
    whistles = audio.detect_whistles_in_file(audio_path)
    rallies = whistles_to_rallies([w.start_sec for w in whistles])
    _write_rallies(out_dir / f"{youtube_id}.json", rallies)
    _write_json(out_dir / f"{youtube_id}.whistles.json", [asdict(w) for w in whistles])
    return f"{len(whistles)} whistles -> {len(rallies)} rallies"


def run_motion_video(video_path: Path, out_dir: Path, youtube_id: str, cache: bool = True) -> str:
    """Motion signal (cached), cuts, occlusions, rallies -> <id>.json, .cuts.json,
    .occlusions.json, .motion.json."""
    sig = motion.compute_motion(video_path, cache=cache)
    cuts = motion.detect_cuts(sig.hist_dist, sig.energy, sig.fps)
    occlusions = motion.detect_blob_occlusions(sig.largest_blob, sig.fps, cuts)
    energy = motion.mask_occlusions(
        motion.mask_cuts(sig.energy, cuts, sig.fps), occlusions, sig.fps
    )
    runs = motion.active_runs(energy, sig.fps)
    rallies = [(s, e) for s, e in runs if MIN_RALLY_SEC <= e - s <= MAX_RALLY_SEC]
    _write_rallies(out_dir / f"{youtube_id}.json", rallies)
    _write_json(
        out_dir / f"{youtube_id}.cuts.json",
        [{"t_sec": round((c + 1) / sig.fps, 2), "hist_dist": round(float(sig.hist_dist[c]), 3)}
         for c in cuts],
    )  # fmt: skip
    level = np.maximum(motion.occlusion_level(sig.largest_blob, sig.fps), 1e-9)
    _write_json(
        out_dir / f"{youtube_id}.occlusions.json",
        [{"start_sec": round(a / sig.fps, 2), "end_sec": round((b + 1) / sig.fps, 2),
          "peak_blob": round(float(sig.largest_blob[a : b + 1].max()), 3),
          "peak_frac": round(float(sig.changed_frac[a : b + 1].max()), 3),
          "blob_ratio": round(float((sig.largest_blob[a : b + 1] / level[a : b + 1]).max()), 1)}
         for a, b in occlusions],
    )  # fmt: skip
    stats = {
        "frames": len(sig.energy) + 1,
        "cuts": len(cuts),
        "occlusions": len(occlusions),
        "occluded_sec": round(sum(b - a + 1 for a, b in occlusions) / sig.fps, 1),
        "active_runs": len(runs),
        "too_short": sum(e - s < MIN_RALLY_SEC for s, e in runs),
        "too_long": sum(e - s > MAX_RALLY_SEC for s, e in runs),
        "rallies": len(rallies),
    }
    _write_json(out_dir / f"{youtube_id}.motion.json", stats)
    return (
        f"{stats['cuts']} cuts, {stats['occlusions']} occlusions ({stats['occluded_sec']} s), "
        f"{len(runs)} active runs ({stats['too_short']} too short, "
        f"{stats['too_long']} too long) -> {len(rallies)} rallies"
    )


def _run_params(detector: str) -> dict:
    common = {"min_rally_sec": MIN_RALLY_SEC, "max_rally_sec": MAX_RALLY_SEC}
    if detector == "whistle":
        return {
            "detector": "whistle",
            "band_hz": list(audio.BAND_HZ),
            "frame": audio.FRAME,
            "hop": audio.HOP,
            **common,
            "dedupe_sec": DEDUPE_SEC,
        }
    return {
        "detector": "motion",
        "fps": motion.FPS,
        "size": list(motion.SIZE),
        "pixel_noise": motion.PIXEL_NOISE,
        "cut_hist_thresh": motion.CUT_HIST_THRESH,
        "cut_spike_ratio": motion.CUT_SPIKE_RATIO,
        "cut_pad": motion.CUT_PAD,
        "occ_ratio": motion.OCC_RATIO,
        "occ_min_frac": motion.OCC_MIN_FRAC,
        "occ_blob_min": motion.OCC_BLOB_MIN,
        "blob_dilate": motion.BLOB_DILATE,
        "occ_max_sec": motion.OCC_MAX_SEC,
        "occ_local_sec": motion.OCC_LOCAL_SEC,
        "occ_pad": motion.OCC_PAD,
        "smooth_sec": motion.SMOOTH_SEC,
        "hi": motion.HI,
        "lo": motion.LO,
        "floor_q": motion.FLOOR_Q,
        "peak_q": motion.PEAK_Q,
        "bridge_sec": motion.BRIDGE_SEC,
        **common,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rally detection for every video in data/raw.")
    parser.add_argument("--raw", default="data/raw", help="dir of <youtube_id>/audio.wav|video.mp4")
    parser.add_argument("--out", default="data/predictions", help="predictions root dir")
    parser.add_argument("--run", required=True, help="run name; writes <out>/<run>/<id>.json")
    parser.add_argument("--videos-doc", help="videos table (default: docs/videos.md)")
    parser.add_argument("--include-held-out", action="store_true", help="also run held-out")
    parser.add_argument("--detector", choices=("whistle", "motion"), default="whistle")
    parser.add_argument("--no-cache", action="store_true", help="motion: recompute the signal")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw)
    out_dir = Path(args.out) / args.run
    try:
        held_out = videos.load_held_out(args.videos_doc, raw_dir, args.include_held_out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    filename = AUDIO_FILENAME if args.detector == "whistle" else VIDEO_FILENAME
    inputs = sorted(raw_dir.glob(f"*/{filename}"))
    if not inputs:
        print(f"error: no */{filename} in {raw_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "_run.json", _run_params(args.detector))
    for path in inputs:
        vid = path.parent.name
        if vid in held_out:
            print(f"{vid}: held out, skipped")
            continue
        if args.detector == "whistle":
            summary = run_whistle_video(path, out_dir, vid)
        else:
            try:
                summary = run_motion_video(path, out_dir, vid, cache=not args.no_cache)
            except motion.MotionError as exc:
                print(f"error: {vid}: {exc}", file=sys.stderr)
                return 1
        print(f"{vid}: {summary}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
