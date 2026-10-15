"""Diagnostic plots for a motion run (matplotlib is a dev-only dependency).

For every scored video with predictions in the run dir, writes to <run>/plots/:

- <id>_timeline.png: stacked ~5 min strips of the whole video. Raw masked energy (light,
  scaled to its own p20..p95) and the smoothed normalized energy the detector thresholds
  (dark), with the hi/lo lines; labels in a lane (green matched, orange overlapped but wrong
  boundaries, red no overlapping prediction; hatched if shorter than SHORT_SEC); predictions
  in a lane below; camera cuts as ticks along the top; occlusions (from <id>.occlusions.json,
  masked like the detector does) as grey bands along the top.
- <id>_aligned.png: normalized energy around every label start and every label end
  (median + IQR), the shape a boundary rule would key on.

Also prints, per video, how many labels have no overlapping prediction and how many of those
are short, against the share of short labels overall. Signals come from the motion cache
(data/raw/<id>/motion_*.npz) using the run's _run.json parameters:

    uv run python -m volley_pipeline.plots --pred ../data/predictions/motion_v0 \
        --labels ../data/labels
"""

import argparse
import json
import math
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from volley_pipeline import motion, videos
from volley_pipeline.diagnose import boundary_diagnosis
from volley_pipeline.eval import DEFAULT_TOLERANCE, Rally, load_rallies, match_rallies
from volley_pipeline.ingest import VIDEO_FILENAME

SHORT_SEC = 4.0
STRIP_SEC = 300.0
ALIGN_WINDOW = (-5.0, 5.0)
Y_RANGE = (-1.0, 2.4)  # normalized energy is clipped to the plot; lanes sit below 0
LABEL_LANE = (-0.55, 0.25)  # (y, height)
PRED_LANE = (-0.9, 0.25)
COLORS = {
    "matched": "#2e9e44",
    "boundary": "#f09a1a",
    "missed": "#d62728",
    "pred": "#1f63b4",
    "occlusion": "#777777",
}


@dataclass
class ShortMisses:
    n_labels: int
    n_short: int
    n_missed: int
    n_missed_short: int


def short_misses(labels: Sequence[Rally], missed: Sequence[Rally], short=SHORT_SEC) -> ShortMisses:
    """Counts of labels / missed labels shorter than `short` s."""
    is_short = lambda r: r[1] - r[0] < short  # noqa: E731
    return ShortMisses(
        n_labels=len(labels),
        n_short=sum(map(is_short, labels)),
        n_missed=len(missed),
        n_missed_short=sum(map(is_short, missed)),
    )


def aligned_profiles(
    norm: np.ndarray,
    fps: float,
    anchors_sec: Sequence[float],
    window: tuple[float, float] = ALIGN_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t, median, q1, q3) of `norm` around each anchor; samples off the ends are ignored.

    Sample k is the frame pair ending at time (k + 1) / fps, as in volley_pipeline.motion.
    """
    offsets = np.arange(round(window[0] * fps), round(window[1] * fps) + 1)
    t = offsets / fps
    rows = np.full((len(anchors_sec), len(offsets)), np.nan)
    for r, a in enumerate(anchors_sec):
        idx = round(a * fps) - 1 + offsets
        ok = (idx >= 0) & (idx < len(norm))
        rows[r, ok] = norm[idx[ok]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> NaN
        q1, med, q3 = np.nanpercentile(rows, [25, 50, 75], axis=0)
    return t, med, q1, q3


def _label_kinds(labels: Sequence[Rally], preds: Sequence[Rally], tolerance: float) -> list[str]:
    matched = {j for _, j in match_rallies(preds, labels, tolerance)}
    missed = set(boundary_diagnosis(labels, preds, tolerance=tolerance).missed)
    return [
        "matched" if j in matched else "missed" if lab in missed else "boundary"
        for j, lab in enumerate(labels)
    ]


def plot_timeline(
    path: Path,
    *,
    raw: np.ndarray,
    norm: np.ndarray,
    fps: float,
    labels: Sequence[Rally],
    kinds: Sequence[str],
    preds: Sequence[Rally],
    cuts_sec: Sequence[float],
    hi: float,
    lo: float,
    title: str,
    strip_sec: float = STRIP_SEC,
    short: float = SHORT_SEC,
    occlusions_sec: Sequence[Rally] = (),
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t = (np.arange(len(norm)) + 1) / fps
    duration = t[-1] if len(t) else 0.0
    n = max(1, math.ceil(duration / strip_sec))
    fig, axes = plt.subplots(n, 1, figsize=(18, 1.9 * n + 0.6), squeeze=False)
    for k, ax in enumerate(axes[:, 0]):
        x0, x1 = k * strip_sec, (k + 1) * strip_sec
        sel = (t >= x0) & (t <= x1)
        ax.plot(t[sel], np.clip(raw[sel], *Y_RANGE), color="#9bb7d4", lw=0.5)
        ax.plot(t[sel], np.clip(norm[sel], *Y_RANGE), color="#18324f", lw=1.0)
        ax.axhline(hi, color="#555", lw=0.7, ls="--")
        ax.axhline(lo, color="#555", lw=0.7, ls=":")
        ax.axhline(0, color="#ccc", lw=0.5)
        for (s, e), kind in zip(labels, kinds, strict=True):
            if e < x0 or s > x1:
                continue
            ax.axvspan(s, e, color=COLORS[kind], alpha=0.10, lw=0)
            ax.broken_barh(
                [(s, e - s)],
                LABEL_LANE,
                facecolor=COLORS[kind],
                edgecolor="black" if e - s < short else "none",
                hatch="////" if e - s < short else None,
                lw=0.6,
            )
        ax.broken_barh(
            [(s, e - s) for s, e in preds if e >= x0 and s <= x1],
            PRED_LANE,
            facecolor=COLORS["pred"],
        )
        for c in cuts_sec:
            if x0 <= c <= x1:
                ax.plot([c, c], [Y_RANGE[1] - 0.25, Y_RANGE[1]], color="#7a3fa0", lw=0.8)
        ax.broken_barh(
            [(s, e - s) for s, e in occlusions_sec if e >= x0 and s <= x1],
            (Y_RANGE[1] - 0.5, 0.5),
            facecolor=COLORS["occlusion"],
            alpha=0.6,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(*Y_RANGE)
        ax.set_yticks([0, lo, hi, 1, 2])
        ax.tick_params(labelsize=7)
        ax.set_xticks(np.arange(x0, x1 + 1, 30))
        ax.set_xticklabels([f"{int(v // 60)}:{int(v % 60):02d}" for v in np.arange(x0, x1 + 1, 30)])
    axes[0, 0].legend(
        handles=[
            Patch(color=COLORS["matched"], label="label: matched"),
            Patch(color=COLORS["boundary"], label="label: overlapped, wrong boundary"),
            Patch(color=COLORS["missed"], label="label: no overlapping prediction"),
            Patch(facecolor="white", edgecolor="black", hatch="////", label=f"label < {short:g} s"),
            Patch(color=COLORS["pred"], label="prediction (lower lane)"),
            Patch(color="#7a3fa0", label="camera cut (top ticks)"),
            Patch(color=COLORS["occlusion"], alpha=0.6, label="occlusion, masked (top band)"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=7,
        ncol=7,
        frameon=False,
    )
    fig.suptitle(
        f"{title}: normalized motion energy (dark: smoothed, scaled p20..p95; light: raw); "
        f"--- hi {hi:g}, ··· lo {lo:g}",
        fontsize=10,
    )
    fig_h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig_h))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_aligned(
    path: Path,
    *,
    norm: np.ndarray,
    fps: float,
    labels: Sequence[Rally],
    hi: float,
    lo: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    panels = [
        ("label start (serve)", [s for s, _ in labels]),
        ("label end (ball dead)", [e for _, e in labels]),
    ]
    for ax, (name, anchors) in zip(axes, panels, strict=True):
        t, med, q1, q3 = aligned_profiles(norm, fps, anchors)
        ax.fill_between(t, q1, q3, color="#9bb7d4", alpha=0.6, lw=0, label="IQR")
        ax.plot(t, med, color="#18324f", lw=1.5, label="median")
        ax.axvline(0, color="black", lw=0.8)
        ax.axhline(hi, color="#555", lw=0.7, ls="--", label=f"hi {hi:g}")
        ax.axhline(lo, color="#555", lw=0.7, ls=":", label=f"lo {lo:g}")
        ax.set_title(f"aligned at {name}, n={len(anchors)}", fontsize=9)
        ax.set_xlabel("seconds from anchor")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("normalized energy")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _run_params(pred_dir: Path) -> dict:
    defaults = {
        "fps": motion.FPS,
        "size": list(motion.SIZE),
        "cut_hist_thresh": motion.CUT_HIST_THRESH,
        "cut_spike_ratio": motion.CUT_SPIKE_RATIO,
        "cut_pad": motion.CUT_PAD,
        "occ_pad": motion.OCC_PAD,
        "smooth_sec": motion.SMOOTH_SEC,
        "hi": motion.HI,
        "lo": motion.LO,
        "floor_q": motion.FLOOR_Q,
        "peak_q": motion.PEAK_Q,
    }
    run_file = pred_dir / "_run.json"
    if run_file.exists():
        defaults.update(json.loads(run_file.read_text()))
    return defaults


def plot_video(
    video_path: Path,
    labels: Sequence[Rally],
    preds: Sequence[Rally],
    out_dir: Path,
    vid: str,
    params: dict,
    tolerance: float = DEFAULT_TOLERANCE,
    short: float = SHORT_SEC,
    occlusions_sec: Sequence[Rally] = (),
) -> tuple[ShortMisses, int]:
    """Write both plots for one video; returns (short-miss counts, number of cuts)."""
    p = params
    sig = motion.compute_motion(video_path, p["fps"], tuple(p["size"]), need_changed=False)
    cuts = motion.detect_cuts(
        sig.hist_dist,
        sig.energy,
        sig.fps,
        hist_thresh=p["cut_hist_thresh"],
        spike_ratio=p["cut_spike_ratio"],
    )
    masked = motion.mask_cuts(sig.energy, cuts, sig.fps, pad=p["cut_pad"])
    occ_runs = [(round(s * sig.fps), round(e * sig.fps) - 1) for s, e in occlusions_sec]
    masked = motion.mask_occlusions(masked, occ_runs, sig.fps, pad=p["occ_pad"])
    scale = {"floor_q": p["floor_q"], "peak_q": p["peak_q"]}
    norm = motion.normalized_energy(masked, sig.fps, smooth_sec=p["smooth_sec"], **scale)
    raw = motion.normalized_energy(masked, sig.fps, smooth_sec=0, **scale)
    if norm is None or raw is None:
        norm = raw = np.zeros(len(masked))
    kinds = _label_kinds(labels, preds, tolerance)
    plot_timeline(
        out_dir / f"{vid}_timeline.png",
        raw=raw,
        norm=norm,
        fps=sig.fps,
        labels=labels,
        kinds=kinds,
        preds=preds,
        cuts_sec=[(c + 1) / sig.fps for c in cuts],
        hi=p["hi"],
        lo=p["lo"],
        title=vid,
        short=short,
        occlusions_sec=occlusions_sec,
    )
    plot_aligned(
        out_dir / f"{vid}_aligned.png",
        norm=norm,
        fps=sig.fps,
        labels=labels,
        hi=p["hi"],
        lo=p["lo"],
        title=vid,
    )
    missed = [lab for lab, k in zip(labels, kinds, strict=True) if k == "missed"]
    return short_misses(labels, missed, short), len(cuts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnostic plots for a motion run.")
    parser.add_argument("--pred", required=True, help="motion run dir (<id>.json + _run.json)")
    parser.add_argument("--labels", default="data/labels", help="dir of labeled <youtube_id>.json")
    parser.add_argument("--raw", help="data/raw (default: next to labels)")
    parser.add_argument("--out", help="output dir (default: <pred>/plots)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="seconds")
    parser.add_argument("--short", type=float, default=SHORT_SEC, help="short rally, seconds")
    parser.add_argument("--videos-doc", help="videos table (default: docs/videos.md)")
    parser.add_argument("--include-held-out", action="store_true", help="also held-out videos")
    args = parser.parse_args(argv)

    pred_dir, label_dir = Path(args.pred), Path(args.labels)
    raw_dir = Path(args.raw) if args.raw else label_dir.parent / "raw"
    out_dir = Path(args.out) if args.out else pred_dir / "plots"
    try:
        held_out = videos.load_held_out(args.videos_doc, label_dir, args.include_held_out)
        params = _run_params(pred_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for label_path in sorted(label_dir.glob("*.json")):
            vid = label_path.stem
            pred_path = pred_dir / f"{vid}.json"
            if vid in held_out or not pred_path.exists():
                continue
            occ_path = pred_dir / f"{vid}.occlusions.json"
            occlusions = (
                [
                    (float(o["start_sec"]), float(o["end_sec"]))
                    for o in json.loads(occ_path.read_text())
                ]
                if occ_path.exists()
                else []
            )
            sm, n_cuts = plot_video(
                raw_dir / vid / VIDEO_FILENAME,
                load_rallies(label_path),
                load_rallies(pred_path),
                out_dir,
                vid,
                params,
                args.tolerance,
                args.short,
                occlusions,
            )
            rows.append((vid, sm, n_cuts))
    except (OSError, ValueError, KeyError, motion.MotionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print(f"error: nothing to plot in {pred_dir}", file=sys.stderr)
        return 1

    s = f"<{args.short:g}s"
    print(f"{'video':<14}{'labels':>7}{s + ' all':>11}{'missed':>8}{s + ' missed':>13}{'cuts':>6}")
    for vid, sm, n_cuts in rows:
        share = sm.n_short / sm.n_labels if sm.n_labels else 0.0
        mshare = sm.n_missed_short / sm.n_missed if sm.n_missed else 0.0
        print(
            f"{vid:<14}{sm.n_labels:>7}{f'{sm.n_short} ({share:.0%})':>11}{sm.n_missed:>8}"
            f"{f'{sm.n_missed_short} ({mshare:.0%})':>13}{n_cuts:>6}"
        )
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
