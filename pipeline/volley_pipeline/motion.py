"""Motion energy from frame differencing, camera-cut detection, and motion-based rallies.

Frames are read through an ffmpeg pipe, sampled at a low rate (default 5 fps) and
downscaled to small grayscale images. Each consecutive pair of frames gives two numbers:

- energy: mean absolute pixel change (0..1), ignoring changes below PIXEL_NOISE
  (compression noise). Players moving, the ball, and camera pans all raise it.
- hist_dist: half the L1 distance between the frames' gray histograms (0..1). Motion and
  pans barely change the histogram; a cut to another shot does.
- changed_frac: share of pixels changing by at least PIXEL_NOISE (0..1). Small players
  moving change a few percent of the frame; someone walking in front of the camera changes
  a large share of it.

A cut is a pair with a large histogram change whose energy is also a lone spike against
the surrounding second. Cuts are reported separately and masked out of the energy, so
broadcast cuts don't count as motion. An occlusion (someone walking in front of the camera)
is a short stretch where changed_frac is far above its local level and covers a large share
of the frame; occlusions are reported and masked the same way. A rally is a stretch where
the smoothed energy, scaled to the video's own floor (p20) and peak (p95), rises above `hi`
and stays above `lo` (hysteresis), with short dips bridged and min/max duration limits.

Pair i ends at frame i + 1, i.e. at time (i + 1) / fps. The per-video signal is cached
next to the video (motion_<fps>fps_<w>x<h>.npz; caches written before changed_frac existed
are re-decoded and rewritten when it's needed). Print one video's signal summary + cuts:

    uv run python -m volley_pipeline.motion ../data/raw/<youtube_id>/video.mp4
"""

import argparse
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from volley_pipeline.eval import Rally

FPS = 5.0
SIZE = (160, 90)  # width, height
PIXEL_NOISE = 8  # gray levels; smaller per-pixel changes count as 0
HIST_BINS = 32
CUT_HIST_THRESH = 0.35
CUT_SPIKE_RATIO = 3.0
CUT_PAD = 1  # pairs masked on each side of a cut
OCC_RATIO = 4.0  # changed_frac this many times its local level...
OCC_MIN_FRAC = 0.15  # ...and at least this share of the frame changing...
OCC_MAX_SEC = 4.0  # ...for at most this long is an occlusion
OCC_LOCAL_SEC = 10.0  # half-window of the local level (rolling median)
OCC_PAD = 2  # pairs masked on each side of an occlusion
SMOOTH_SEC = 1.0
HI = 0.5  # start threshold, as a fraction of the floor -> peak range
LO = 0.3  # continue threshold
FLOOR_Q = 20
PEAK_Q = 95
BRIDGE_SEC = 1.0
# same defaults as segment.MIN_RALLY_SEC / MAX_RALLY_SEC (segment imports this module)
MIN_RALLY_SEC = 1.0
MAX_RALLY_SEC = 45.0


class MotionError(Exception):
    """Frames could not be read from the video."""


@dataclass
class MotionSignal:
    energy: np.ndarray  # float32, one value per consecutive frame pair
    hist_dist: np.ndarray  # float32, same length
    fps: float
    changed_frac: np.ndarray | None = None  # float32, same length; None if loaded without it


def iter_gray_frames(
    video_path: str | Path, fps: float = FPS, size: tuple[int, int] = SIZE
) -> Iterator[np.ndarray]:
    """Yield (h, w) uint8 grayscale frames sampled at `fps`, scaled to `size` (w, h)."""
    video_path = Path(video_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MotionError("ffmpeg not found on PATH (install it: brew install ffmpeg)")
    if not video_path.is_file():
        raise MotionError(f"Video file not found: {video_path}")
    w, h = size
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(video_path),
        "-an", "-vf", f"fps={fps:g},scale={w}:{h},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]  # fmt: skip
    frame_bytes = w * h
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
        proc.wait()
        if proc.returncode != 0:
            assert proc.stderr is not None
            err = proc.stderr.read().decode(errors="replace")
            tail = "\n".join(err.strip().splitlines()[-5:])
            raise MotionError(f"ffmpeg failed to read frames: {tail}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def _hist(frame: np.ndarray) -> np.ndarray:
    counts = np.bincount((frame >> 3).ravel(), minlength=HIST_BINS)  # 256 / 32 = 8 levels
    return counts / frame.size


def pair_signals(frames: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(energy, hist_dist, changed_frac) per consecutive frame pair; all length n_frames - 1."""
    energy: list[float] = []
    hist_dist: list[float] = []
    changed: list[float] = []
    prev: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    for frame in frames:
        cur = frame.astype(np.int16)
        cur_hist = _hist(frame)
        if prev is not None:
            diff = np.abs(cur - prev)
            diff[diff < PIXEL_NOISE] = 0
            energy.append(float(diff.mean()) / 255)
            hist_dist.append(0.5 * float(np.abs(cur_hist - prev_hist).sum()))
            changed.append(float(np.count_nonzero(diff)) / diff.size)
        prev, prev_hist = cur, cur_hist
    return tuple(np.asarray(x, dtype=np.float32) for x in (energy, hist_dist, changed))


def frame_signals(frames: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(energy, hist_dist) per consecutive frame pair; both length n_frames - 1."""
    energy, hist_dist, _ = pair_signals(frames)
    return energy, hist_dist


def _local_median(x: np.ndarray, i: int, half: int, keep: np.ndarray | None = None) -> float:
    lo, hi = max(0, i - half), min(len(x), i + half + 1)
    window = x[lo:hi] if keep is None else x[lo:hi][keep[lo:hi]]
    return float(np.median(window)) if len(window) else 0.0


def detect_cuts(
    hist_dist: np.ndarray,
    energy: np.ndarray,
    fps: float = FPS,
    *,
    hist_thresh: float = CUT_HIST_THRESH,
    spike_ratio: float = CUT_SPIKE_RATIO,
) -> list[int]:
    """Indices of frame pairs that are camera cuts: big histogram change + lone energy spike."""
    half = max(1, round(fps))
    cuts = []
    for i in np.flatnonzero(hist_dist > hist_thresh):
        if energy[i] >= spike_ratio * max(_local_median(energy, int(i), half), 1e-6):
            cuts.append(int(i))
    return cuts


def mask_cuts(
    energy: np.ndarray, cuts: Iterable[int], fps: float = FPS, pad: int = CUT_PAD
) -> np.ndarray:
    """Copy of energy with each cut (± pad pairs) replaced by the median of the unmasked ±1 s."""
    out = np.asarray(energy, dtype=np.float32).copy()
    keep = np.ones(len(out), dtype=bool)
    for c in cuts:
        keep[max(0, c - pad) : c + pad + 1] = False
    half = max(1, round(fps))
    for i in np.flatnonzero(~keep):
        out[i] = _local_median(energy, int(i), half, keep)
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs where mask is True."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def occlusion_level(
    changed_frac: np.ndarray, fps: float = FPS, local_sec: float = OCC_LOCAL_SEC
) -> np.ndarray:
    """Local level of changed_frac: rolling median over ±local_sec, floored at the video median."""
    x = np.asarray(changed_frac, dtype=np.float64)
    if len(x) == 0:
        return x
    local = median_filter(x, size=2 * max(1, round(local_sec * fps)) + 1, mode="nearest")
    return np.maximum(local, np.median(x))


def detect_occlusions(
    changed_frac: np.ndarray,
    fps: float = FPS,
    cuts: Iterable[int] = (),
    *,
    ratio: float = OCC_RATIO,
    min_frac: float = OCC_MIN_FRAC,
    max_sec: float = OCC_MAX_SEC,
    local_sec: float = OCC_LOCAL_SEC,
    cut_pad: int = CUT_PAD,
) -> list[tuple[int, int]]:
    """Inclusive pair-index runs where changed_frac is > ratio x its local level and
    >= min_frac, lasting at most max_sec; runs touching a cut (± cut_pad) are left to the cut."""
    x = np.asarray(changed_frac, dtype=np.float64)
    if len(x) == 0:
        return []
    level = occlusion_level(x, fps, local_sec)
    high = (x > ratio * np.maximum(level, 1e-9)) & (x >= min_frac)
    near_cut = np.zeros(len(x), dtype=bool)
    for c in cuts:
        near_cut[max(0, c - cut_pad) : c + cut_pad + 1] = True
    return [
        (a, b)
        for a, b in _runs(high)
        if (b - a + 1) / fps <= max_sec and not near_cut[a : b + 1].any()
    ]


def mask_occlusions(
    energy: np.ndarray, runs: Iterable[tuple[int, int]], fps: float = FPS, pad: int = OCC_PAD
) -> np.ndarray:
    """Copy of energy with each run (± pad pairs) replaced by the median of the unmasked samples
    in the second before and the second after it."""
    out = np.asarray(energy, dtype=np.float32).copy()
    n = len(out)
    keep = np.ones(n, dtype=bool)
    for a, b in runs:
        keep[max(0, a - pad) : b + pad + 1] = False
    half = max(1, round(fps))
    for a, b in _runs(~keep):
        context = np.r_[np.arange(max(0, a - half), a), np.arange(b + 1, min(n, b + 1 + half))]
        context = context[keep[context]]
        out[a : b + 1] = float(np.median(energy[context])) if len(context) else 0.0
    return out


def normalized_energy(
    energy: np.ndarray,
    fps: float = FPS,
    *,
    smooth_sec: float = SMOOTH_SEC,
    floor_q: float = FLOOR_Q,
    peak_q: float = PEAK_Q,
) -> np.ndarray | None:
    """Smoothed energy scaled so the video's floor_q -> 0 and peak_q -> 1; None if flat/empty."""
    x = np.asarray(energy, dtype=np.float64)
    if len(x) == 0:
        return None
    s = uniform_filter1d(x, size=max(1, round(smooth_sec * fps)), mode="nearest")
    floor, peak = np.percentile(s, [floor_q, peak_q])
    if peak - floor <= 1e-12:
        return None
    return (s - floor) / (peak - floor)


def active_runs(
    energy: np.ndarray,
    fps: float = FPS,
    *,
    smooth_sec: float = SMOOTH_SEC,
    hi: float = HI,
    lo: float = LO,
    floor_q: float = FLOOR_Q,
    peak_q: float = PEAK_Q,
    bridge_sec: float = BRIDGE_SEC,
) -> list[Rally]:
    """Active stretches (start_sec, end_sec) of the smoothed signal, before duration limits."""
    norm = normalized_energy(energy, fps, smooth_sec=smooth_sec, floor_q=floor_q, peak_q=peak_q)
    if norm is None:
        return []

    runs: list[tuple[int, int]] = []  # inclusive pair indices
    i, n = 0, len(norm)
    while i < n:
        if norm[i] > lo:
            j = i
            while j + 1 < n and norm[j + 1] > lo:
                j += 1
            if norm[i : j + 1].max() > hi:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1

    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) / fps < bridge_sec:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [((a + 1) / fps, (b + 1) / fps) for a, b in merged]


def motion_to_rallies(
    energy: np.ndarray,
    fps: float = FPS,
    *,
    min_rally: float = MIN_RALLY_SEC,
    max_rally: float = MAX_RALLY_SEC,
    **kwargs,
) -> list[Rally]:
    """Active stretches lasting min_rally..max_rally s; kwargs go to `active_runs`."""
    return [
        (s, e) for s, e in active_runs(energy, fps, **kwargs) if min_rally <= e - s <= max_rally
    ]


def cache_path(video_path: str | Path, fps: float = FPS, size: tuple[int, int] = SIZE) -> Path:
    return Path(video_path).parent / f"motion_{fps:g}fps_{size[0]}x{size[1]}.npz"


def compute_motion(
    video_path: str | Path,
    fps: float = FPS,
    size: tuple[int, int] = SIZE,
    cache: bool = True,
    need_changed: bool = True,
) -> MotionSignal:
    """Signal for a video, read from / written to the cache next to it.

    A cache without changed_frac (written before it existed) is re-decoded and rewritten when
    need_changed, else loaded with changed_frac=None.
    """
    path = cache_path(video_path, fps, size)
    if cache and path.is_file():
        with np.load(path) as z:
            changed = z["changed_frac"] if "changed_frac" in z.files else None
            if changed is not None or not need_changed:
                return MotionSignal(z["energy"], z["hist_dist"], float(z["fps"]), changed)
    energy, hist_dist, changed = pair_signals(iter_gray_frames(video_path, fps, size))
    sig = MotionSignal(energy, hist_dist, fps, changed)
    if cache:
        tmp = path.with_name(path.stem + ".tmp.npz")
        with open(tmp, "wb") as f:
            np.savez(
                f, energy=energy, hist_dist=hist_dist, changed_frac=changed, fps=np.float64(fps)
            )
        tmp.replace(path)
    return sig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Motion energy + camera cuts for one video.")
    parser.add_argument("video", help="path to video.mp4")
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--no-cache", action="store_true", help="recompute, don't write cache")
    args = parser.parse_args(argv)
    try:
        sig = compute_motion(args.video, args.fps, cache=not args.no_cache)
    except MotionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cuts = detect_cuts(sig.hist_dist, sig.energy, sig.fps)
    occlusions = detect_occlusions(sig.changed_frac, sig.fps, cuts)
    energy = mask_occlusions(mask_cuts(sig.energy, cuts, sig.fps), occlusions, sig.fps)
    rallies = motion_to_rallies(energy, sig.fps)
    q = np.percentile(sig.energy, [20, 50, 95]) if len(sig.energy) else [0, 0, 0]
    print(
        f"{len(sig.energy) + 1} frames at {sig.fps:g} fps; energy p20/p50/p95 "
        f"{q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}; {len(cuts)} cuts; {len(occlusions)} occlusions; "
        f"{len(rallies)} rallies"
    )
    for c in cuts:
        print(f"cut at {(c + 1) / sig.fps:8.1f}s  hist_dist {sig.hist_dist[c]:.2f}")
    for a, b in occlusions:
        print(
            f"occlusion {a / sig.fps:8.1f}s - {(b + 1) / sig.fps:8.1f}s  "
            f"peak changed {sig.changed_frac[a : b + 1].max():.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
