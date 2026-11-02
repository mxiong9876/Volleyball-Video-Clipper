import shutil
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile

from volley_pipeline import detectors, segment
from volley_pipeline.eval import load_rallies


def test_current_detector_is_registered():
    assert detectors.CURRENT in detectors.REGISTRY


def test_unknown_detector_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown detector 'nope'"):
        detectors.detect("nope", tmp_path)


def test_detect_dispatches_by_name(monkeypatch, tmp_path):
    seen = []

    def fake(media_dir):
        seen.append(media_dir)
        return [(1.0, 2.0)]

    monkeypatch.setitem(detectors.REGISTRY, "fake_v1", fake)
    assert detectors.detect("fake_v1", str(tmp_path)) == [(1.0, 2.0)]
    assert seen == [tmp_path]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_motion_v1_matches_the_cli_runner(tmp_path):
    """4 s black, 3 s moving pattern, 4 s black: same rallies as run_motion_video's file."""
    media = tmp_path / "media"
    media.mkdir()
    # fmt: off
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=black:s=160x90:r=30:d=4",
         "-f", "lavfi", "-i", "testsrc2=s=160x90:r=30:d=3",
         "-f", "lavfi", "-i", "color=c=black:s=160x90:r=30:d=4",
         "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p",
         str(media / "video.mp4")],
        check=True,
    )
    # fmt: on
    rallies = detectors.detect("motion_v1", media)
    assert rallies

    out = tmp_path / "out"
    out.mkdir()
    segment.run_motion_video(media / "video.mp4", out, "vid")
    written = load_rallies(out / "vid.json")
    assert [(round(s, 2), round(e, 2)) for s, e in rallies] == written


def test_whistle_v0_reads_audio(tmp_path):
    silence = np.zeros(22050, dtype=np.float32)
    wavfile.write(tmp_path / "audio.wav", 22050, silence)
    assert detectors.detect("whistle_v0", tmp_path) == []
