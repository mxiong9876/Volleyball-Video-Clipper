"""Referee whistle detection: sustained narrow-band energy in roughly 2-4 kHz.

A whistle is a strong, tonal peak inside the band: the band's loudest bin stands far above
the band's median bin (crowd roar and cheering are broadband, so their peak stays close to
the median), and the band holds a large share of the frame's energy (rejects tones under
loud speech). Frames passing both tests are grouped into runs; runs of plausible whistle
length are returned. Plain functions over arrays; `detect_whistles_in_file` reads a WAV.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile

BAND_HZ = (2000.0, 4000.0)
FRAME = 1024  # 46 ms at 22050 Hz, ~21.5 Hz bins
HOP = 256  # 11.6 ms at 22050 Hz
_CHUNK_FRAMES = 4096  # STFT this many frames at a time to bound memory on long matches
_TINY = 1e-12


@dataclass(frozen=True)
class Whistle:
    start_sec: float
    end_sec: float
    peak_hz: float  # median in-band peak frequency over the run
    strength_db: float  # median in-band peak-to-median ratio over the run


@dataclass(frozen=True)
class Activity:
    """Per-frame whistle features; `times` are frame centers in seconds."""

    times: np.ndarray
    peak_db: np.ndarray  # in-band peak bin vs in-band median bin, dB
    band_frac: np.ndarray  # in-band energy / total frame energy
    peak_hz: np.ndarray
    hop_sec: float


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float32 in [-1, 1]; returns (samples, sample_rate)."""
    sr, data = wavfile.read(path)
    if data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float32) / float(-np.iinfo(data.dtype).min)
    else:
        x = data.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, int(sr)


def whistle_activity(
    samples: np.ndarray,
    sr: int,
    band: tuple[float, float] = BAND_HZ,
    frame: int = FRAME,
    hop: int = HOP,
) -> Activity:
    """Short-time spectrum features for whistle detection, one value per hop."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("samples must be mono (1-D)")
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n_frames = 1 + (len(x) - frame) // hop
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        raise ValueError(f"band {band} has no FFT bins at sr={sr}, frame={frame}")
    band_freqs = freqs[in_band]
    window = np.hanning(frame).astype(np.float32)

    peak_db = np.empty(n_frames, dtype=np.float32)
    band_frac = np.empty(n_frames, dtype=np.float32)
    peak_hz = np.empty(n_frames, dtype=np.float32)
    for f0 in range(0, n_frames, _CHUNK_FRAMES):
        f1 = min(f0 + _CHUNK_FRAMES, n_frames)
        seg = x[f0 * hop : (f1 - 1) * hop + frame]
        frames = sliding_window_view(seg, frame)[::hop] * window
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        bp = power[:, in_band]
        peak_idx = bp.argmax(axis=1)
        peak = bp[np.arange(len(bp)), peak_idx]
        peak_db[f0:f1] = 10 * np.log10((peak + _TINY) / (np.median(bp, axis=1) + _TINY))
        band_frac[f0:f1] = bp.sum(axis=1) / (power.sum(axis=1) + _TINY)
        peak_hz[f0:f1] = band_freqs[peak_idx]

    times = (np.arange(n_frames) * hop + frame / 2) / sr
    return Activity(times, peak_db, band_frac, peak_hz, hop / sr)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def detect_whistles(
    samples: np.ndarray,
    sr: int,
    *,
    band: tuple[float, float] = BAND_HZ,
    min_peak_db: float = 15.0,
    min_band_frac: float = 0.25,
    min_dur: float = 0.15,
    max_dur: float = 3.0,
    merge_gap: float = 0.1,
) -> list[Whistle]:
    """Whistles as time spans; runs closer than `merge_gap` s merge before duration filters."""
    act = whistle_activity(samples, sr, band=band)
    mask = (act.peak_db >= min_peak_db) & (act.band_frac >= min_band_frac)
    runs = _runs(mask)

    merged: list[list[int]] = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) * act.hop_sec <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    half = act.hop_sec / 2
    whistles = []
    for s, e in merged:
        start, end = act.times[s] - half, act.times[e - 1] + half
        if not min_dur <= end - start <= max_dur:
            continue
        on = mask[s:e]
        whistles.append(
            Whistle(
                start_sec=round(float(start), 3),
                end_sec=round(float(end), 3),
                peak_hz=round(float(np.median(act.peak_hz[s:e][on])), 1),
                strength_db=round(float(np.median(act.peak_db[s:e][on])), 1),
            )
        )
    return whistles


def detect_whistles_in_file(path: str | Path, **kwargs) -> list[Whistle]:
    samples, sr = load_wav(path)
    return detect_whistles(samples, sr, **kwargs)
