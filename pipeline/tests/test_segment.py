import json
import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile

from volley_pipeline import segment
from volley_pipeline.eval import load_rallies

SR = 22050


def test_alternating_whistles_pair_into_rallies():
    times = [10.0, 18.0, 40.0, 52.0, 80.0, 85.0]
    assert segment.whistles_to_rallies(times) == [(10.0, 18.0), (40.0, 52.0), (80.0, 85.0)]


def test_unsorted_input():
    assert segment.whistles_to_rallies([40.0, 10.0, 52.0, 18.0]) == [(10.0, 18.0), (40.0, 52.0)]


def test_double_blow_is_one_whistle():
    assert segment.whistles_to_rallies([10.0, 10.4, 18.0, 18.9]) == [(10.0, 18.0)]


def test_long_gap_drops_stray_start():
    # 5.0 has no partner within max_rally, so 100.0 becomes the serve whistle
    assert segment.whistles_to_rallies([5.0, 100.0, 110.0]) == [(100.0, 110.0)]


def test_trailing_unpaired_whistle_is_dropped():
    assert segment.whistles_to_rallies([10.0, 18.0, 30.0]) == [(10.0, 18.0)]


def test_empty_and_single():
    assert segment.whistles_to_rallies([]) == []
    assert segment.whistles_to_rallies([3.0]) == []


def test_dedupe_whistles():
    assert segment.dedupe_whistles([3.0, 1.0, 1.5, 2.6]) == [1.0, 2.6]


def _whistle_wav(path, at_secs, seconds=30):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, seconds * SR)
    t = np.arange(int(0.5 * SR)) / SR
    for at in at_secs:
        i = int(at * SR)
        x[i : i + len(t)] += 0.2 * np.sin(2 * np.pi * 3000 * t)
    path.parent.mkdir(parents=True)
    wavfile.write(path, SR, (x * 32767).astype(np.int16))


def test_cli_writes_label_format_and_skips_held_out(tmp_path, capsys):
    raw = tmp_path / "raw"
    _whistle_wav(raw / "aaaaaaaaaaa" / "audio.wav", [2.0, 10.0, 20.0, 25.0])
    _whistle_wav(raw / "bbbbbbbbbbb" / "audio.wav", [2.0, 10.0])
    doc = tmp_path / "videos.md"
    doc.write_text(
        "| ID | Split |\n|---|---|\n| aaaaaaaaaaa | tune |\n| bbbbbbbbbbb | held-out |\n"
    )
    out = tmp_path / "pred"

    code = segment.main(
        ["--raw", str(raw), "--out", str(out), "--run", "r1", "--videos-doc", str(doc)]
    )
    assert code == 0
    run_dir = out / "r1"
    assert not (run_dir / "bbbbbbbbbbb.json").exists()
    rallies = load_rallies(run_dir / "aaaaaaaaaaa.json")
    assert [(round(s), round(e)) for s, e in rallies] == [(2, 10), (20, 25)]
    whistles = json.loads((run_dir / "aaaaaaaaaaa.whistles.json").read_text())
    assert len(whistles) == 4 and {"start_sec", "end_sec", "peak_hz"} <= whistles[0].keys()
    assert "held out" in capsys.readouterr().out

    code = segment.main(
        ["--raw", str(raw), "--out", str(out), "--run", "r2", "--videos-doc", str(doc),
         "--include-held-out"]
    )  # fmt: skip
    assert code == 0
    [(s, e)] = load_rallies(out / "r2" / "bbbbbbbbbbb.json")
    assert abs(s - 2.0) < 0.05 and abs(e - 10.0) < 0.05


def test_cli_errors_without_audio(tmp_path):
    (tmp_path / "raw").mkdir()
    code = segment.main(["--raw", str(tmp_path / "raw"), "--out", str(tmp_path), "--run", "x"])
    assert code == 1


def _motion_clip(path):
    """4 s black, 3 s moving test pattern, 4 s black (30 fps), generated with ffmpeg."""
    path.parent.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=30:duration=4",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=3",
         "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=30:duration=4",
         "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_cli_motion_detector_writes_rallies_cuts_and_skips_held_out(tmp_path, capsys):
    raw = tmp_path / "raw"
    _motion_clip(raw / "aaaaaaaaaaa" / "video.mp4")
    _motion_clip(raw / "bbbbbbbbbbb" / "video.mp4")
    doc = tmp_path / "videos.md"
    doc.write_text(
        "| ID | Split |\n|---|---|\n| aaaaaaaaaaa | tune |\n| bbbbbbbbbbb | held-out |\n"
    )
    out = tmp_path / "pred"

    code = segment.main(
        ["--raw", str(raw), "--out", str(out), "--run", "m1", "--videos-doc", str(doc),
         "--detector", "motion"]
    )  # fmt: skip
    assert code == 0
    run_dir = out / "m1"
    assert not (run_dir / "bbbbbbbbbbb.json").exists()
    [(s, e)] = load_rallies(run_dir / "aaaaaaaaaaa.json")
    assert abs(s - 4.0) <= 1.0 and abs(e - 7.0) <= 1.0
    cuts = [c["t_sec"] for c in json.loads((run_dir / "aaaaaaaaaaa.cuts.json").read_text())]
    assert any(abs(t - 4.0) <= 0.4 for t in cuts)
    assert json.loads((run_dir / "_run.json").read_text())["detector"] == "motion"
    assert json.loads((run_dir / "aaaaaaaaaaa.motion.json").read_text())["rallies"] == 1
    assert (raw / "aaaaaaaaaaa" / "motion_5fps_160x90.npz").is_file()
    assert "held out" in capsys.readouterr().out


def test_cli_motion_errors_without_video(tmp_path):
    (tmp_path / "raw").mkdir()
    code = segment.main(
        ["--raw", str(tmp_path / "raw"), "--out", str(tmp_path), "--run", "x",
         "--detector", "motion"]
    )  # fmt: skip
    assert code == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_cli_motion_detector_writes_occlusions(tmp_path):
    raw = tmp_path / "raw"
    _motion_clip(raw / "aaaaaaaaaaa" / "video.mp4")
    out = tmp_path / "pred"
    code = segment.main(
        ["--raw", str(raw), "--out", str(out), "--run", "m1", "--detector", "motion"]
    )  # fmt: skip
    assert code == 0
    run_dir = out / "m1"
    # the pattern starts and ends with cuts, so it stays motion, not an occlusion
    assert json.loads((run_dir / "aaaaaaaaaaa.occlusions.json").read_text()) == []
    stats = json.loads((run_dir / "aaaaaaaaaaa.motion.json").read_text())
    assert stats["occlusions"] == 0 and stats["rallies"] == 1
    assert json.loads((run_dir / "_run.json").read_text())["occ_min_frac"] == 0.15
