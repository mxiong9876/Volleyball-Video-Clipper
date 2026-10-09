import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

from volley_pipeline import audio

SR = 22050


def noise(seconds, std=0.05, seed=0):
    return np.random.default_rng(seed).normal(0, std, int(seconds * SR)).astype(np.float32)


def tone(seconds, hz, amp=0.2):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def trill(seconds, lo=3000, hi=3300, rate=25, amp=0.2):
    """Frequency-modulated tone, like a pea whistle's warble."""
    t = np.arange(int(seconds * SR)) / SR
    inst_hz = (lo + hi) / 2 + (hi - lo) / 2 * np.sin(2 * np.pi * rate * t)
    phase = 2 * np.pi * np.cumsum(inst_hz) / SR
    return (amp * np.sin(phase)).astype(np.float32)


def place(background, clip, at_sec):
    out = background.copy()
    i = int(at_sec * SR)
    out[i : i + len(clip)] += clip
    return out


def test_detects_tones_in_noise_at_known_times():
    x = noise(10)
    for at in (1.0, 4.0, 7.5):
        x = place(x, tone(0.5, 3000), at)
    whistles = audio.detect_whistles(x, SR)
    assert [w.start_sec for w in whistles] == pytest.approx([1.0, 4.0, 7.5], abs=0.05)
    assert [w.end_sec for w in whistles] == pytest.approx([1.5, 4.5, 8.0], abs=0.05)
    assert all(abs(w.peak_hz - 3000) < 25 for w in whistles)


def test_detects_trilled_whistle():
    x = place(noise(4), trill(0.8), 2.0)
    [w] = audio.detect_whistles(x, SR)
    assert w.start_sec == pytest.approx(2.0, abs=0.05)
    assert 2950 < w.peak_hz < 3350


def test_amplitude_scaling_does_not_change_result():
    x = place(noise(4), tone(0.5, 3200), 1.0)
    base = audio.detect_whistles(x, SR)
    assert len(base) == 1
    assert audio.detect_whistles(x * 0.05, SR) == base
    assert audio.detect_whistles(x * 4, SR) == base


def test_pure_noise_has_no_whistles():
    assert audio.detect_whistles(noise(10, seed=3), SR) == []


def test_ignores_out_of_band_tone():
    x = place(noise(4), tone(1.0, 1000), 1.0)
    assert audio.detect_whistles(x, SR) == []


def test_ignores_too_short_blip():
    x = place(noise(4), tone(0.05, 3000), 1.0)
    assert audio.detect_whistles(x, SR) == []


def test_ignores_too_long_tone():
    x = place(noise(10), tone(6.0, 3000), 1.0)
    assert audio.detect_whistles(x, SR) == []


def test_ignores_loud_broadband_burst():
    """Crowd roar: much louder than the background but not tonal."""
    x = place(noise(4), noise(1.0, std=0.5, seed=7), 1.0)
    assert audio.detect_whistles(x, SR) == []


def test_ignores_in_band_noise_burst():
    """Noise confined to 2-4 kHz is in band but not narrow-band."""
    sos = butter(8, [2000, 4000], btype="bandpass", fs=SR, output="sos")
    burst = sosfilt(sos, noise(1.0, std=0.5, seed=9)).astype(np.float32)
    x = place(noise(4), burst, 1.0)
    assert audio.detect_whistles(x, SR) == []


def test_close_runs_merge():
    """A whistle with a 50 ms dropout is one whistle."""
    x = noise(4)
    x = place(x, tone(0.3, 3000), 1.0)
    x = place(x, tone(0.3, 3000), 1.35)
    [w] = audio.detect_whistles(x, SR)
    assert w.start_sec == pytest.approx(1.0, abs=0.05)
    assert w.end_sec == pytest.approx(1.65, abs=0.05)


def test_shorter_than_one_frame_is_fine():
    assert audio.detect_whistles(np.zeros(100, dtype=np.float32), SR) == []


def test_load_wav_int16_mono(tmp_path):
    x = place(noise(2), tone(0.4, 3000), 0.5)
    p = tmp_path / "a.wav"
    wavfile.write(p, SR, (x * 32767).astype(np.int16))
    samples, sr = audio.load_wav(p)
    assert sr == SR and samples.dtype == np.float32 and samples.ndim == 1
    assert samples == pytest.approx(x, abs=1e-3)
    [w] = audio.detect_whistles_in_file(p)
    assert w.start_sec == pytest.approx(0.5, abs=0.05)


def test_load_wav_downmixes_stereo(tmp_path):
    left = np.full(1000, 0.5, dtype=np.float32)
    right = np.full(1000, -0.25, dtype=np.float32)
    p = tmp_path / "s.wav"
    wavfile.write(p, SR, np.stack([left, right], axis=1))
    samples, _ = audio.load_wav(p)
    assert samples.ndim == 1
    assert samples == pytest.approx(np.full(1000, 0.125), abs=1e-6)
