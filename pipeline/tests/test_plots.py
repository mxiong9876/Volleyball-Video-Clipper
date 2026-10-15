import json

import numpy as np
import pytest

from volley_pipeline import motion, plots

FPS = 5.0


def test_aligned_profiles_step_at_anchor():
    norm = np.zeros(200)
    anchors = [10.0, 20.0, 30.0]
    for a in anchors:
        k = round(a * FPS) - 1  # sample k ends at (k + 1) / fps
        norm[k : k + 10] = 1.0
    t, med, q1, q3 = plots.aligned_profiles(norm, FPS, anchors, window=(-1.0, 1.0))
    assert t[0] == pytest.approx(-1.0) and t[-1] == pytest.approx(1.0)
    assert np.all(med[t < 0] == 0) and np.all(med[t >= 0] == 1)
    assert np.array_equal(q1, med) and np.array_equal(q3, med)


def test_aligned_profiles_ignores_samples_off_the_ends():
    norm = np.arange(20, dtype=float)
    t, med, _, _ = plots.aligned_profiles(norm, FPS, [0.2, 3.8], window=(-1.0, 1.0))
    # before the first sample only the second anchor contributes; after the last, only the first
    assert med[0] == norm[round(3.8 * FPS) - 1 - 5]
    assert med[-1] == norm[round(0.2 * FPS) - 1 + 5]
    t, med, _, _ = plots.aligned_profiles(norm, FPS, [100.0], window=(-1.0, 1.0))
    assert np.isnan(med).all()


def test_short_misses():
    labels = [(0.0, 2.0), (10.0, 20.0), (30.0, 33.0), (40.0, 50.0)]
    sm = plots.short_misses(labels, [(0.0, 2.0), (40.0, 50.0)], short=4.0)
    assert (sm.n_labels, sm.n_short, sm.n_missed, sm.n_missed_short) == (4, 2, 2, 1)


def test_cli_writes_both_plots(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    vid = "aaaaaaaaaaa"
    raw, labels, pred = tmp_path / "raw" / vid, tmp_path / "labels", tmp_path / "pred"
    for d in (raw, labels, pred):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    energy = (0.01 + rng.normal(0, 0.002, 600)).astype(np.float32)  # 2 min at 5 fps
    energy[100:150] = 0.1
    hist = np.zeros_like(energy)
    hist[300], energy[300] = 0.9, 0.9  # one cut
    video = raw / "video.mp4"  # never read: the cache is used
    np.savez(motion.cache_path(video), energy=energy, hist_dist=hist, fps=np.float64(FPS))
    rows = lambda rs: json.dumps([{"start_sec": s, "end_sec": e} for s, e in rs])  # noqa: E731
    (labels / f"{vid}.json").write_text(rows([(20.0, 30.0), (80.0, 82.0)]))
    (pred / f"{vid}.json").write_text(rows([(20.2, 30.0)]))

    code = plots.main(["--pred", str(pred), "--labels", str(labels), "--raw", str(raw.parent)])
    assert code == 0
    assert (pred / "plots" / f"{vid}_timeline.png").stat().st_size > 0
    assert (pred / "plots" / f"{vid}_aligned.png").stat().st_size > 0
    out = capsys.readouterr().out
    assert vid in out and "1 (100%)" in out  # the one missed label is short
