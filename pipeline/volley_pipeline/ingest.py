"""Ingestion: YouTube URL -> metadata, video.mp4 and mono WAV audio.

Plain functions over URLs and paths; no DB or web imports. Run the whole stage with:

    uv run python -m volley_pipeline.ingest <url> --out data/raw
"""

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp
from yt_dlp.networking.exceptions import HTTPError, TransportError

MAX_DURATION_SEC = 3 * 3600
AUDIO_SAMPLE_RATE = 22050
VIDEO_FILENAME = "video.mp4"
AUDIO_FILENAME = "audio.wav"
# 720p is plenty for motion energy and scoreboard OCR, and keeps files small.
VIDEO_FORMAT = "bv*[height<=720]+ba/b[height<=720]/b"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com"}
_PATH_PREFIXES = ("shorts", "embed", "live", "v")


class IngestError(Exception):
    """Base class; messages are written to be shown to users as-is."""


class InvalidURLError(IngestError):
    pass


class VideoUnavailableError(IngestError):
    pass


class LookupTimeoutError(IngestError):
    pass


class VideoTooLongError(IngestError):
    pass


class DownloadError(IngestError):
    pass


class AudioExtractionError(IngestError):
    pass


def parse_youtube_id(url: str) -> str:
    """Extract the 11-char video id from a YouTube URL, without any network access."""
    invalid = InvalidURLError(f"Not a valid YouTube video URL: {url!r}")
    candidate = url.strip()
    if "://" not in candidate:
        candidate = "https://" + candidate
    try:
        parsed = urlparse(candidate)
    except ValueError:
        raise invalid from None
    if parsed.scheme not in ("http", "https"):
        raise invalid
    host = (parsed.hostname or "").lower().removeprefix("www.")
    parts = [p for p in parsed.path.split("/") if p]

    video_id = None
    if host == "youtu.be" and parts:
        video_id = parts[0]
    elif host in _YOUTUBE_HOSTS:
        if parts == ["watch"]:
            video_id = (parse_qs(parsed.query).get("v") or [None])[0]
        elif len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
            video_id = parts[1]

    if not video_id or not _ID_RE.match(video_id):
        raise invalid
    return video_id


def _is_network_error(exc: BaseException | None) -> bool:
    """Walk yt-dlp's wrapped exceptions looking for a transport-level failure."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, HTTPError):
            return False  # the server answered; that's not "unreachable"
        if isinstance(exc, TransportError | socket.timeout | TimeoutError | ConnectionError):
            return True
        inner = getattr(exc, "exc_info", None)
        nxt = inner[1] if inner and inner[1] is not exc else None
        exc = nxt or getattr(exc, "cause", None) or exc.__cause__ or exc.__context__
    return False


def _clean_ytdlp_message(msg: str) -> str:
    # "ERROR: [youtube] abcdefghijk: Private video. Sign in..." -> "Private video. Sign in..."
    return re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?([\w-]{11}:\s*)?", "", str(msg)).strip()


def fetch_metadata(url: str, timeout: float = 10) -> dict:
    """Look up title and duration without downloading.

    Returns {"youtube_id", "title", "duration_sec"}. Raises VideoUnavailableError
    (private/removed/etc.) or LookupTimeoutError (YouTube unreachable).
    """
    youtube_id = parse_youtube_id(url)
    opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": timeout,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        if _is_network_error(exc):
            raise LookupTimeoutError(
                "Couldn't reach YouTube to look up the video; try again shortly"
            ) from exc
        raise VideoUnavailableError(
            f"YouTube video unavailable: {_clean_ytdlp_message(str(exc))}"
        ) from exc
    if not info or info.get("duration") is None:
        raise VideoUnavailableError("YouTube didn't report a duration (is this a live stream?)")
    return {
        "youtube_id": info.get("id") or youtube_id,
        "title": info.get("title") or youtube_id,
        "duration_sec": int(info["duration"]),
    }


def _fmt_duration(seconds: int) -> str:
    h, m = divmod(seconds // 60, 60)
    if not h:
        return f"{m}m"
    return f"{h}h {m:02d}m" if m else f"{h}h"


def check_duration(duration_sec: int, max_sec: int = MAX_DURATION_SEC) -> None:
    if duration_sec > max_sec:
        raise VideoTooLongError(
            f"Video is {_fmt_duration(duration_sec)} long; the limit is {_fmt_duration(max_sec)}"
        )


def _remove_partial_files(out_dir: Path) -> None:
    """Delete everything yt-dlp may leave behind from an interrupted run.

    That's any `video.*` other than the final merged video.mp4: .part, .ytdl,
    per-format intermediates (video.f137.mp4) and merge temps (video.temp.mp4).
    """
    for path in out_dir.glob("video.*"):
        if path.name != VIDEO_FILENAME and path.is_file():
            path.unlink()


class _ProgressTracker:
    """Turns yt-dlp hook events (one file per format) into one overall 0-100 percent."""

    def __init__(self, on_progress: Callable[[float], None]):
        self.on_progress = on_progress
        self.downloaded: dict[str, int] = {}
        self.last_pct = 0.0

    def __call__(self, d: dict) -> None:
        if d.get("status") not in ("downloading", "finished"):
            return
        name = d.get("filename", "")
        self.downloaded[name] = d.get("downloaded_bytes") or d.get("total_bytes") or 0

        info = d.get("info_dict") or {}
        sizes = [
            f.get("filesize") or f.get("filesize_approx") or 0
            for f in info.get("requested_formats") or []
        ]
        total = sum(sizes) if sizes and all(sizes) else 0
        if not total:
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = self.downloaded[name]
        else:
            done = sum(self.downloaded.values())
        if not total:
            return
        pct = min(100.0, 100.0 * done / total)
        if pct > self.last_pct:  # monotonic; per-format files would otherwise reset to 0
            self.last_pct = pct
            self.on_progress(pct)


def download_video(
    url: str, out_dir: str | Path, on_progress: Callable[[float], None] | None = None
) -> Path:
    """Download to out_dir/video.mp4 (<=720p). Skips if a complete video.mp4 exists.

    yt-dlp merges into a temp file and renames at the end, so an existing non-empty
    video.mp4 is complete. Partial leftovers are deleted before retrying.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / VIDEO_FILENAME
    if final.is_file() and final.stat().st_size > 0:
        return final
    final.unlink(missing_ok=True)
    _remove_partial_files(out_dir)

    opts = {
        "format": VIDEO_FORMAT,
        "outtmpl": str(out_dir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        # Single-file formats that aren't mp4 (e.g. webm) still end up as video.mp4.
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_ProgressTracker(on_progress)] if on_progress else [],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"Download failed: {_clean_ytdlp_message(str(exc))}") from exc
    if not final.is_file():
        raise DownloadError(f"Download finished but {final.name} was not produced")
    return final


def extract_audio(
    video_path: str | Path, out_path: str | Path, sample_rate: int = AUDIO_SAMPLE_RATE
) -> Path:
    """Extract mono 16-bit PCM WAV at sample_rate. Writes to a temp file, then renames."""
    video_path, out_path = Path(video_path), Path(out_path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioExtractionError("ffmpeg not found on PATH (install it: brew install ffmpeg)")
    if not video_path.is_file():
        raise AudioExtractionError(f"Video file not found: {video_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        str(tmp),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise AudioExtractionError(f"ffmpeg failed to extract audio: {tail}")
    tmp.replace(out_path)
    return out_path


def ingest(url: str, raw_dir: str | Path, max_duration_sec: int = MAX_DURATION_SEC) -> dict:
    """Whole stage: lookup -> duration check -> download -> audio. Returns metadata + paths."""
    meta = fetch_metadata(url)
    check_duration(meta["duration_sec"], max_duration_sec)
    out_dir = Path(raw_dir) / meta["youtube_id"]
    video = download_video(url, out_dir)
    audio = extract_audio(video, out_dir / AUDIO_FILENAME)
    return {**meta, "video_path": str(video), "audio_path": str(audio)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a YouTube video and extract audio.")
    parser.add_argument("url")
    parser.add_argument("--out", default="data/raw", help="raw media dir (default: data/raw)")
    args = parser.parse_args(argv)
    try:
        result = ingest(args.url, args.out)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
