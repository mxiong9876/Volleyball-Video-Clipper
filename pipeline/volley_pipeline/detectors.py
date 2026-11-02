"""Named rally detectors. The app runs CURRENT; changing detector means changing that one line.

A name identifies one detector behaviour, and it is stored as the rallies' source in the DB. When
a detector's parameters change, register it under a new name instead of editing an existing entry.
"""

from collections.abc import Callable
from pathlib import Path

from volley_pipeline import audio, segment
from volley_pipeline.eval import Rally
from volley_pipeline.ingest import AUDIO_FILENAME, VIDEO_FILENAME


def _motion_v1(media_dir: Path) -> list[Rally]:
    return segment.detect_motion(media_dir / VIDEO_FILENAME).rallies


def _whistle_v0(media_dir: Path) -> list[Rally]:
    whistles = audio.detect_whistles_in_file(media_dir / AUDIO_FILENAME)
    return segment.whistles_to_rallies([w.start_sec for w in whistles])


REGISTRY: dict[str, Callable[[Path], list[Rally]]] = {
    "motion_v1": _motion_v1,
    "whistle_v0": _whistle_v0,
}
CURRENT = "motion_v1"


def detect(name: str, media_dir: str | Path) -> list[Rally]:
    """Run a registered detector on data/raw/<id>/ (video.mp4 and audio.wav)."""
    if name not in REGISTRY:
        raise ValueError(f"unknown detector {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](Path(media_dir))
