"""Why did the detector miss? Bucket every unmatched label and prediction of a run.

Labels the eval doesn't match (at `tolerance`) are classified by which whistles were
detected near them. A serve whistle is a whistle onset in `serve_window` around the label
start (the server has seconds after the whistle to serve); an end whistle is one in
`end_window` around the label end (blown just after the ball is dead):

- no_whistle: neither found
- serve_missed: only the end whistle found
- end_missed: only the serve whistle found
- offset_only: both found and a prediction spans them; it's wrong only by the serve delay /
  end lag, so a better boundary rule (not a better whistle detector) would fix it
- mispaired: both found but no prediction spans them (segmentation paired other whistles)

Unmatched predictions are dead_time (overlap no label) or overlap (cover a label, wrong
boundaries). `chance` is the share of random windows of the serve window's length that hold
a whistle (Poisson, from the whistle rate): serve-window hit rates near it mean the "hits"
are coincidence, i.e. no real referee whistle. Reads a segment run dir:

    uv run python -m volley_pipeline.diagnose --pred ../data/predictions/<run> \
        --labels ../data/labels
"""

import argparse
import bisect
import json
import math
import sys
import wave
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median, quantiles

from volley_pipeline import videos
from volley_pipeline.eval import DEFAULT_TOLERANCE, Rally, load_rallies, match_rallies

SERVE_WINDOW = (-10.0, 1.0)  # whistle onset - label start
END_WINDOW = (-1.0, 4.0)  # whistle onset - label end
LABEL_BUCKETS = ("matched", "offset_only", "mispaired", "serve_missed", "end_missed", "no_whistle")
PRED_BUCKETS = ("dead_time", "overlap")


@dataclass
class Diagnosis:
    labels: Counter = field(default_factory=Counter)
    preds: Counter = field(default_factory=Counter)
    n_labels: int = 0
    n_whistles: int = 0
    duration_sec: float = 0.0
    serve_hits: int = 0  # labels with a whistle in the serve window
    chance: float = 0.0
    # nearest-whistle offsets (whistle - label) for labels that have one
    serve_offsets: list[float] = field(default_factory=list)
    end_offsets: list[float] = field(default_factory=list)


def _in_window(times: Sequence[float], t: float, window: tuple[float, float]) -> list[float]:
    lo = bisect.bisect_left(times, t + window[0])
    hi = bisect.bisect_right(times, t + window[1])
    return list(times[lo:hi])


def _overlaps(a: Rally, b: Rally) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def diagnose_video(
    labels: Sequence[Rally],
    predicted: Sequence[Rally],
    whistle_times: Sequence[float],
    *,
    duration_sec: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    serve_window: tuple[float, float] = SERVE_WINDOW,
    end_window: tuple[float, float] = END_WINDOW,
) -> Diagnosis:
    times = sorted(whistle_times)
    if duration_sec is None:
        ends = [e for _, e in labels] + [e for _, e in predicted] + times
        duration_sec = max(ends, default=0.0)
    pairs = match_rallies(predicted, labels, tolerance)
    matched_labels = {j for _, j in pairs}
    matched_preds = {i for i, _ in pairs}

    d = Diagnosis(n_labels=len(labels), n_whistles=len(times), duration_sec=duration_sec)
    rate = len(times) / duration_sec if duration_sec > 0 else 0.0
    d.chance = 1 - math.exp(-rate * (serve_window[1] - serve_window[0]))

    for j, (start, end) in enumerate(labels):
        serve = _in_window(times, start, serve_window)
        stop = _in_window(times, end, end_window)
        if serve:
            d.serve_hits += 1
            d.serve_offsets.append(serve[-1] - start)  # last whistle before the serve
        if stop:
            d.end_offsets.append(stop[0] - end)  # first whistle after the ball is dead
        if j in matched_labels:
            bucket = "matched"
        elif serve and stop:
            spans = any(
                start + serve_window[0] <= ps <= start + serve_window[1]
                and end + end_window[0] <= pe <= end + end_window[1]
                for ps, pe in predicted
            )
            bucket = "offset_only" if spans else "mispaired"
        elif stop:
            bucket = "serve_missed"
        elif serve:
            bucket = "end_missed"
        else:
            bucket = "no_whistle"
        d.labels[bucket] += 1

    for i, p in enumerate(predicted):
        if i not in matched_preds:
            d.preds["overlap" if any(_overlaps(p, lab) for lab in labels) else "dead_time"] += 1
    return d


def _merge(ds: Sequence[Diagnosis]) -> Diagnosis:
    total = Diagnosis()
    for d in ds:
        total.labels += d.labels
        total.preds += d.preds
        total.n_labels += d.n_labels
        total.n_whistles += d.n_whistles
        total.duration_sec += d.duration_sec
        total.serve_hits += d.serve_hits
        total.serve_offsets += d.serve_offsets
        total.end_offsets += d.end_offsets
    if total.duration_sec > 0:
        rate = total.n_whistles / total.duration_sec
        total.chance = 1 - math.exp(-rate * (SERVE_WINDOW[1] - SERVE_WINDOW[0]))
    return total


def _dist(xs: list[float]) -> str:
    if not xs:
        return "-"
    if len(xs) < 2:
        return f"{xs[0]:+.1f}"
    q1, _, q3 = quantiles(xs, n=4)
    return f"{median(xs):+.1f} [{q1:+.1f},{q3:+.1f}]"


def format_diagnosis(per_video: dict[str, Diagnosis]) -> str:
    cols = [("labels", 7), *((b, len(b) + 2) for b in LABEL_BUCKETS)]
    cols += [(b, len(b) + 2) for b in PRED_BUCKETS]
    header = f"{'video':<14}" + "".join(f"{name:>{w}}" for name, w in cols)
    lines = ["labels: matched + misses by bucket | unmatched predictions", header]
    lines.append("-" * len(header))
    rows = [*per_video.items(), ("overall", _merge(list(per_video.values())))]
    for name, d in rows:
        if name == "overall":
            lines.append("-" * len(header))
        vals = [d.n_labels, *(d.labels[b] for b in LABEL_BUCKETS)]
        vals += [d.preds[b] for b in PRED_BUCKETS]
        lines.append(
            f"{name:<14}" + "".join(f"{v:>{w}}" for v, (_, w) in zip(vals, cols, strict=True))
        )

    header2 = (
        f"{'video':<14}{'whistles':>9}{'/min':>6}{'serve hit':>11}{'chance':>8}"
        f"{'serve whistle − start':>26}{'end whistle − end':>24}"
    )
    lines += ["", "whistles near labels (offsets: median [IQR], s)", header2]
    lines.append("-" * len(header2))
    for name, d in rows:
        if name == "overall":
            lines.append("-" * len(header2))
        per_min = d.n_whistles / d.duration_sec * 60 if d.duration_sec else 0.0
        hit = d.serve_hits / d.n_labels if d.n_labels else 0.0
        lines.append(
            f"{name:<14}{d.n_whistles:>9}{per_min:>6.1f}{hit:>11.0%}{d.chance:>8.0%}"
            f"{_dist(d.serve_offsets):>26}{_dist(d.end_offsets):>24}"
        )
    return "\n".join(lines)


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except (OSError, wave.Error, EOFError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bucket a run's misses by nearby whistles.")
    parser.add_argument("--pred", required=True, help="segment run dir (<id>.json + whistles)")
    parser.add_argument("--labels", default="data/labels", help="dir of labeled <youtube_id>.json")
    parser.add_argument("--raw", help="data/raw, for video durations (default: next to labels)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="seconds")
    parser.add_argument("--videos-doc", help="videos table (default: docs/videos.md)")
    parser.add_argument("--include-held-out", action="store_true", help="also held-out videos")
    args = parser.parse_args(argv)

    pred_dir, label_dir = Path(args.pred), Path(args.labels)
    raw_dir = Path(args.raw) if args.raw else label_dir.parent / "raw"
    per_video: dict[str, Diagnosis] = {}
    try:
        held_out = videos.load_held_out(args.videos_doc, label_dir, args.include_held_out)
        for label_path in sorted(label_dir.glob("*.json")):
            vid = label_path.stem
            if vid in held_out:
                continue
            pred_path, whistle_path = pred_dir / f"{vid}.json", pred_dir / f"{vid}.whistles.json"
            if not pred_path.exists() or not whistle_path.exists():
                print(f"warning: no predictions/whistles for {vid}; skipped", file=sys.stderr)
                continue
            whistles = [float(w["start_sec"]) for w in json.loads(whistle_path.read_text())]
            per_video[vid] = diagnose_video(
                load_rallies(label_path),
                load_rallies(pred_path),
                whistles,
                duration_sec=_wav_duration(raw_dir / vid / "audio.wav"),
                tolerance=args.tolerance,
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not per_video:
        print(f"error: nothing to diagnose in {pred_dir}", file=sys.stderr)
        return 1
    print(f"tolerance ±{args.tolerance:g}s; serve window {SERVE_WINDOW}, end window {END_WINDOW}")
    print(format_diagnosis(per_video))
    return 0


if __name__ == "__main__":
    sys.exit(main())
