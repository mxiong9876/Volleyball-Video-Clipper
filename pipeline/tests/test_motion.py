import shutil
import subprocess

import numpy as np
import pytest

from volley_pipeline import motion

FPS = 5.0
H, W = 90, 160


def gray(level):
    return np.full((H, W), level, dtype=np.uint8)


def square_at(x, level=40, size=20):
    """Dark background with a bright square whose left edge is at column x."""
    f = gray(level)
    f[30 : 30 + size, x : x + size] = 220
    return f


def texture(seed):
    return np.random.default_rng(seed).integers(0, 256, (H, W), dtype=np.uint8)


def plateaus(seconds, spans, base=0.01, high=0.1, seed=0):
    """Low noisy energy with `high` plateaus over the given (start, end) second spans.

    Sample k is the pair ending at frame k + 1, i.e. time (k + 1) / FPS.
    """
    n = int(seconds * FPS)
    x = base + np.random.default_rng(seed).normal(0, base * 0.2, n)
    t = (np.arange(n) + 1) / FPS
    for s, e in spans:
        x[(t >= s) & (t <= e)] = high
    return x.astype(np.float32)


# --- frame_signals -------------------------------------------------------------------------


def test_identical_frames_have_zero_energy():
    energy, hist = motion.frame_signals([gray(100)] * 4)
    assert len(energy) == len(hist) == 3
    assert energy.max() == 0 and hist.max() == 0


def test_moving_square_is_motion_not_a_cut():
    energy, hist = motion.frame_signals([square_at(x) for x in (10, 20, 30, 40)])
    assert (energy > 0).all()
    assert hist.max() < 0.01  # same pixels, moved: histogram unchanged


def test_scene_swap_changes_histogram():
    _, hist = motion.frame_signals([gray(30), gray(200)])
    assert hist[0] == pytest.approx(1.0)


def test_small_changes_below_pixel_noise_ignored():
    energy, _ = motion.frame_signals([gray(100), gray(100 + motion.PIXEL_NOISE - 1)])
    assert energy[0] == 0


# --- cuts ----------------------------------------------------------------------------------


def test_detect_single_cut_at_known_index():
    # 3 s of a slowly moving square on dark, then a cut to a bright texture shot
    frames = [square_at(10 + x) for x in range(15)]
    frames += [np.maximum(texture(1), 128) for _ in range(15)]
    energy, hist = motion.frame_signals(frames)
    assert motion.detect_cuts(hist, energy, FPS) == [14]  # pair (14, 15)


def test_sustained_fast_motion_is_not_a_cut():
    # big jumps every frame: lots of energy, but neither a histogram change nor a lone spike
    frames = [square_at(x) for x in [10, 120, 10, 120, 10, 120, 10, 120, 10, 120, 10, 120]]
    energy, hist = motion.frame_signals(frames)
    assert motion.detect_cuts(hist, energy, FPS) == []


def test_repeated_scene_swaps_are_not_lone_spikes():
    frames = [gray(30), gray(200)] * 8  # every pair is a full change: no spike vs neighbors
    energy, hist = motion.frame_signals(frames)
    assert motion.detect_cuts(hist, energy, FPS) == []


def test_mask_cuts_removes_spike_and_keeps_others():
    energy = np.full(30, 0.02, dtype=np.float32)
    energy[::2] = 0.03
    energy[15] = 0.9
    out = motion.mask_cuts(energy, [15], FPS, pad=1)
    assert out.max() < 0.05
    untouched = np.r_[0:14, 17:30]
    assert np.array_equal(out[untouched], energy[untouched])
    assert energy[15] == pytest.approx(0.9)  # input not modified


# --- motion_to_rallies ---------------------------------------------------------------------


def test_plateaus_become_rallies_at_known_times():
    spans = [(10.0, 18.0), (40.0, 52.0), (80.0, 85.0)]
    rallies = motion.motion_to_rallies(plateaus(120, spans), FPS)
    assert len(rallies) == 3
    for (s, e), (ls, le) in zip(rallies, spans, strict=True):
        assert abs(s - ls) <= 1 / FPS + 1e-6
        assert abs(e - le) <= 1 / FPS + 1e-6


def test_short_dip_is_bridged():
    x = plateaus(60, [(10.0, 20.0)])
    t = (np.arange(len(x)) + 1) / FPS
    x[(t >= 15.0) & (t <= 15.4)] = 0.01  # 0.4 s pause mid-rally
    [(s, e)] = motion.motion_to_rallies(x, FPS, smooth_sec=0.2)  # little smoothing
    assert s == pytest.approx(10.0, abs=0.3) and e == pytest.approx(20.0, abs=0.3)


def test_short_blip_dropped_by_min_rally():
    x = plateaus(60, [(10.0, 20.0), (40.0, 40.4)])
    rallies = motion.motion_to_rallies(x, FPS, smooth_sec=0.2)
    assert len(rallies) == 1


def test_overlong_stretch_dropped_by_max_rally():
    x = plateaus(200, [(10.0, 20.0), (60.0, 130.0), (150.0, 158.0)])
    rallies = motion.motion_to_rallies(x, FPS, max_rally=45.0)
    assert [round(s) for s, _ in rallies] == [10, 150]
    assert len(motion.active_runs(x, FPS)) == 3


def test_scale_independent():
    x = plateaus(120, [(10.0, 18.0), (40.0, 52.0)])
    assert motion.motion_to_rallies(x * 10, FPS) == motion.motion_to_rallies(x, FPS)


def test_flat_or_empty_signal_has_no_rallies():
    assert motion.motion_to_rallies(np.zeros(100, dtype=np.float32), FPS) == []
    assert motion.motion_to_rallies(np.zeros(0, dtype=np.float32), FPS) == []


# --- ffmpeg reader + cache (generated clip, no stored media) -------------------------------

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def make_clip(path, seconds=2):
    """Moving test pattern for `seconds`, then a cut to solid white for 1 s (30 fps)."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", "color=c=white:size=320x180:rate=30:duration=1",
         "-filter_complex", "[0][1]concat=n=2:v=1:a=0",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_iter_gray_frames_shape_and_rate(tmp_path):
    clip = make_clip(tmp_path / "video.mp4")
    frames = list(motion.iter_gray_frames(clip, fps=5, size=(160, 90)))
    assert 14 <= len(frames) <= 16  # 3 s at 5 fps
    assert frames[0].shape == (90, 160) and frames[0].dtype == np.uint8
    assert frames[-1].mean() > 240  # white tail


@needs_ffmpeg
def test_compute_motion_finds_cut_and_caches(tmp_path):
    clip = make_clip(tmp_path / "video.mp4")
    sig = motion.compute_motion(clip)
    cache = motion.cache_path(clip)
    assert cache.name == "motion_5fps_160x90.npz" and cache.is_file()
    cuts = motion.detect_cuts(sig.hist_dist, sig.energy, sig.fps)
    assert len(cuts) == 1 and abs((cuts[0] + 1) / sig.fps - 2.0) <= 0.4

    clip.unlink()  # second call must come from the cache
    again = motion.compute_motion(clip)
    assert np.array_equal(again.energy, sig.energy) and again.fps == 5.0
    with pytest.raises(motion.MotionError):
        motion.compute_motion(clip, fps=2)  # different key -> recompute -> no video


def test_missing_video_raises(tmp_path):
    with pytest.raises(motion.MotionError):
        list(motion.iter_gray_frames(tmp_path / "nope.mp4"))


def test_normalized_energy_scales_to_floor_and_peak():
    x = plateaus(120, [(10.0, 18.0), (40.0, 52.0)])
    norm = motion.normalized_energy(x, FPS)
    assert np.percentile(norm, motion.FLOOR_Q) == pytest.approx(0, abs=1e-9)
    assert np.percentile(norm, motion.PEAK_Q) == pytest.approx(1, abs=1e-9)
    assert np.allclose(motion.normalized_energy(x * 10, FPS), norm)


def test_normalized_energy_none_when_flat_or_empty():
    assert motion.normalized_energy(np.zeros(50, dtype=np.float32), FPS) is None
    assert motion.normalized_energy(np.zeros(0, dtype=np.float32), FPS) is None


# --- changed fraction + occlusions -----------------------------------------------------------

PLAYERS = [(20, 15), (50, 40), (80, 20), (110, 45), (135, 25), (65, 60)]  # (x, y) of small players


def court_frame(k, players=PLAYERS, size=(14, 6), step=3):
    """Flat background with small bright players shifting `step` px back and forth each frame."""
    f = gray(60)
    h, w = size
    for x, y in players:
        x += step * (k % 2)
        f[y : y + h, x : x + w] = 200
    return f


def court_clip(seconds, blob_at=None, blob_sec=2.0, blob_w=70):
    """Court frames; optionally a full-height dark blob crossing left to right from `blob_at` s."""
    frames = []
    for k in range(int(seconds * FPS)):
        f = court_frame(k)
        t = k / FPS
        if blob_at is not None and blob_at <= t < blob_at + blob_sec:
            x = round(-blob_w + (t - blob_at) / blob_sec * (W + blob_w))
            f[:, max(0, x) : max(0, x + blob_w)] = 10
        frames.append(f)
    return frames


def test_changed_fraction_counts_pixels_above_noise():
    _, _, changed = motion.pair_signals([gray(100)] * 3)
    assert changed.max() == 0
    _, _, changed = motion.pair_signals([square_at(10), square_at(60)])  # moved clear of itself
    assert changed[0] == pytest.approx(2 * 400 / (H * W))
    _, _, changed = motion.pair_signals([gray(100), gray(100 + motion.PIXEL_NOISE - 1)])
    assert changed[0] == 0


def test_frame_signals_matches_pair_signals():
    frames = [square_at(x) for x in (10, 20, 30, 40)]
    energy, hist = motion.frame_signals(frames)
    e2, h2, changed = motion.pair_signals(frames)
    assert np.array_equal(energy, e2) and np.array_equal(hist, h2) and len(changed) == 3


def test_large_blob_crossing_is_flagged():
    _, _, changed = motion.pair_signals(court_clip(40, blob_at=20.0))
    occ = motion.detect_occlusions(changed, FPS)
    assert len(occ) == 1
    (a, b) = occ[0]
    assert 19.0 <= a / FPS <= 21.0 and 21.0 <= (b + 1) / FPS <= 23.0


def test_small_moving_players_are_not_flagged():
    _, _, changed = motion.pair_signals(court_clip(40))
    assert changed.max() > 0 and changed.max() < motion.OCC_MIN_FRAC
    assert motion.detect_occlusions(changed, FPS) == []


def test_burst_of_near_side_players_is_not_flagged():
    # two far players all along; four bigger near-side players sprint for 2 s at t = 20 s
    far, near = PLAYERS[:2], [(10, 50), (50, 55), (90, 50), (130, 55)]
    frames = []
    for k in range(int(40 * FPS)):
        f = court_frame(k, far)
        if 20.0 <= k / FPS < 22.0:
            f2 = court_frame(k, near, size=(30, 12), step=5)
            f = np.where(f2 != 60, f2, f)
        frames.append(f)
    _, _, changed = motion.pair_signals(frames)
    level = motion.occlusion_level(changed, FPS)
    burst = slice(int(20.2 * FPS), int(21.8 * FPS))
    assert (changed[burst] > motion.OCC_RATIO * level[burst]).all()  # far above local level...
    assert changed.max() < motion.OCC_MIN_FRAC  # ...but a small share of the frame
    assert motion.detect_occlusions(changed, FPS) == []


def test_long_high_stretch_and_cuts_are_not_occlusions():
    x = np.full(int(60 * FPS), 0.03, dtype=np.float32)
    x[100:110] = 0.4  # 2 s: occlusion
    x[200:230] = 0.4  # 6 s: too long
    assert motion.detect_occlusions(x, FPS) == [(100, 109)]
    assert motion.detect_occlusions(x, FPS, cuts=[105]) == []


def test_mask_occlusions_fills_from_surroundings():
    energy = np.full(100, 0.02, dtype=np.float32)
    energy[40:50] = 0.5
    out = motion.mask_occlusions(energy, [(40, 49)], FPS, pad=2)
    assert np.allclose(out[38:52], 0.02)
    assert np.array_equal(out[:38], energy[:38]) and np.array_equal(out[52:], energy[52:])
    assert energy[45] == pytest.approx(0.5)  # input not modified


@needs_ffmpeg
def test_cache_stores_changed_frac_and_upgrades_old_caches(tmp_path):
    clip = make_clip(tmp_path / "video.mp4")
    sig = motion.compute_motion(clip)
    cache = motion.cache_path(clip)
    with np.load(cache) as z:
        assert np.array_equal(z["changed_frac"], sig.changed_frac)
    # rewrite as a pre-changed_frac cache
    np.savez(cache, energy=sig.energy, hist_dist=sig.hist_dist, fps=np.float64(5.0))
    old = motion.compute_motion(clip, need_changed=False)
    assert old.changed_frac is None and np.array_equal(old.energy, sig.energy)
    upgraded = motion.compute_motion(clip)
    assert np.array_equal(upgraded.changed_frac, sig.changed_frac)
    with np.load(cache) as z:
        assert "changed_frac" in z.files
