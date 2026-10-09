import json

import numpy as np
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
