import shutil
import subprocess
import wave
from pathlib import Path

import pytest
import yt_dlp
from yt_dlp.networking.exceptions import TransportError

from volley_pipeline import ingest

VID = "dQw4w9WgXcQ"


# --- parse_youtube_id ------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s&list=PL123",
        f"http://m.youtube.com/watch?feature=share&v={VID}",
        f"https://music.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/live/{VID}?feature=share",
        f"youtube.com/watch?v={VID}",
        f"  https://youtu.be/{VID}  ",
    ],
)
def test_parse_youtube_id_valid(url):
    assert ingest.parse_youtube_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        f"https://vimeo.com/{VID}",
        f"https://notyoutube.com/watch?v={VID}",
        "https://www.youtube.com/watch?v=tooshort",
        f"https://www.youtube.com/watch?v={VID}X",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/@somechannel",
        "https://www.youtube.com/playlist?list=PL123",
        f"ftp://youtube.com/watch?v={VID}",
    ],
)
def test_parse_youtube_id_invalid(url):
    with pytest.raises(ingest.InvalidURLError, match="Not a valid YouTube video URL"):
        ingest.parse_youtube_id(url)


# --- fake yt-dlp -----------------------------------------------------------


class FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL. Configure via class attributes per test."""

    info: dict | None = None
    error: Exception | None = None
    on_download = None  # callable(fake, url)
    instances: list["FakeYDL"] = []

    def __init__(self, opts):
        self.opts = opts
        FakeYDL.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        if FakeYDL.error:
            raise FakeYDL.error
        return FakeYDL.info

    def download(self, urls):
        if FakeYDL.error:
            raise FakeYDL.error
        if FakeYDL.on_download:
            FakeYDL.on_download(self, urls[0])
        return 0


@pytest.fixture
def fake_ydl(monkeypatch):
    FakeYDL.info, FakeYDL.error, FakeYDL.on_download = None, None, None
    FakeYDL.instances = []
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return FakeYDL


URL = f"https://www.youtube.com/watch?v={VID}"


# --- fetch_metadata --------------------------------------------------------


def test_fetch_metadata_ok(fake_ydl):
    fake_ydl.info = {"id": VID, "title": "Finals Set 1", "duration": 5432.6}
    meta = ingest.fetch_metadata(URL, timeout=7)
    assert meta == {"youtube_id": VID, "title": "Finals Set 1", "duration_sec": 5432}
    opts = fake_ydl.instances[0].opts
    assert opts["socket_timeout"] == 7
    assert opts["skip_download"] is True


def test_fetch_metadata_rejects_invalid_url_without_network(fake_ydl):
    with pytest.raises(ingest.InvalidURLError):
        ingest.fetch_metadata("https://example.com/video")
    assert fake_ydl.instances == []


def test_fetch_metadata_private_video(fake_ydl):
    fake_ydl.error = yt_dlp.utils.DownloadError(
        f"ERROR: [youtube] {VID}: Private video. Sign in if you've been granted access"
    )
    with pytest.raises(ingest.VideoUnavailableError) as exc_info:
        ingest.fetch_metadata(URL)
    assert str(exc_info.value) == (
        "YouTube video unavailable: Private video. Sign in if you've been granted access"
    )


def test_fetch_metadata_network_timeout(fake_ydl):
    cause = TransportError("The read operation timed out")
    fake_ydl.error = yt_dlp.utils.DownloadError(
        "ERROR: Unable to download webpage", exc_info=(type(cause), cause, None)
    )
    with pytest.raises(ingest.LookupTimeoutError, match="Couldn't reach YouTube"):
        ingest.fetch_metadata(URL)


def test_fetch_metadata_live_stream_without_duration(fake_ydl):
    fake_ydl.info = {"id": VID, "title": "Live now", "duration": None}
    with pytest.raises(ingest.VideoUnavailableError, match="duration"):
        ingest.fetch_metadata(URL)


# --- check_duration --------------------------------------------------------


def test_check_duration_at_limit_ok():
    ingest.check_duration(3 * 3600)


def test_check_duration_one_second_over_reads_differently():
    with pytest.raises(ingest.VideoTooLongError) as exc_info:
        ingest.check_duration(3 * 3600 + 1)
    assert str(exc_info.value) == "Video is 3h 01m long; the limit is 3h"


def test_check_duration_over_limit():
    with pytest.raises(ingest.VideoTooLongError) as exc_info:
        ingest.check_duration(3 * 3600 + 12 * 60)
    assert str(exc_info.value) == "Video is 3h 12m long; the limit is 3h"


# --- download_video --------------------------------------------------------


def fake_merge_download(seen_files):
    """Simulates yt-dlp downloading two formats, firing hooks, then merging to video.mp4."""

    def run(fake, url):
        out_dir = Path(fake.opts["outtmpl"]).parent
        seen_files.extend(sorted(p.name for p in out_dir.iterdir()))
        info = {"requested_formats": [{"filesize": 800}, {"filesize": 200}]}
        for hook in fake.opts["progress_hooks"]:
            for name, done in [("video.f137.mp4", 400), ("video.f137.mp4", 800)]:
                hook({"status": "downloading", "filename": name, "downloaded_bytes": done,
                      "info_dict": info})  # fmt: skip
            for name, done in [("video.f140.m4a", 0), ("video.f140.m4a", 200)]:
                hook({"status": "downloading", "filename": name, "downloaded_bytes": done,
                      "info_dict": info})  # fmt: skip
        (out_dir / "video.mp4").write_bytes(b"merged")

    return run


def test_download_video_writes_mp4_and_reports_progress(fake_ydl, tmp_path):
    seen, progress = [], []
    fake_ydl.on_download = fake_merge_download(seen)
    path = ingest.download_video(URL, tmp_path / VID, on_progress=progress.append)

    assert path == tmp_path / VID / "video.mp4"
    assert path.read_bytes() == b"merged"
    assert progress == [40.0, 80.0, 100.0]  # monotonic across the two format files
    opts = fake_ydl.instances[0].opts
    assert opts["merge_output_format"] == "mp4"
    assert opts["outtmpl"] == str(tmp_path / VID / "video.%(ext)s")


def test_download_video_skips_when_complete_mp4_exists(fake_ydl, tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"done before")
    path = ingest.download_video(URL, tmp_path)
    assert path.read_bytes() == b"done before"
    assert fake_ydl.instances == []


def test_download_video_does_not_skip_on_part_file(fake_ydl, tmp_path):
    (tmp_path / "video.mp4.part").write_bytes(b"half")
    (tmp_path / "video.f137.mp4").write_bytes(b"intermediate")
    (tmp_path / "video.f140.m4a.ytdl").write_bytes(b"state")
    (tmp_path / "audio.wav").write_bytes(b"keep me")
    seen = []
    fake_ydl.on_download = fake_merge_download(seen)

    path = ingest.download_video(URL, tmp_path)

    assert len(fake_ydl.instances) == 1  # did download
    assert seen == ["audio.wav"]  # leftovers were gone before yt-dlp ran
    assert path.read_bytes() == b"merged"
    assert (tmp_path / "audio.wav").read_bytes() == b"keep me"


def test_download_video_redownloads_empty_mp4(fake_ydl, tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"")
    fake_ydl.on_download = fake_merge_download([])
    assert ingest.download_video(URL, tmp_path).read_bytes() == b"merged"


def test_download_video_error(fake_ydl, tmp_path):
    fake_ydl.error = yt_dlp.utils.DownloadError("ERROR: HTTP Error 403: Forbidden")
    with pytest.raises(ingest.DownloadError, match="Download failed: HTTP Error 403"):
        ingest.download_video(URL, tmp_path)


def test_download_video_missing_output(fake_ydl, tmp_path):
    with pytest.raises(ingest.DownloadError, match="not produced"):
        ingest.download_video(URL, tmp_path)


# --- extract_audio (real ffmpeg on a generated clip) -------------------------


@pytest.fixture
def tiny_clip(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.fail("ffmpeg not found on PATH; install it (brew install ffmpeg)")
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=44100:duration=1",
         "-ac", "2", "-shortest", str(clip)],
        check=True,
    )  # fmt: skip
    return clip


def test_extract_audio_mono_22k(tiny_clip, tmp_path):
    out = ingest.extract_audio(tiny_clip, tmp_path / "out" / "audio.wav")
    assert out == tmp_path / "out" / "audio.wav"
    assert not (tmp_path / "out" / "audio.tmp.wav").exists()
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 22050
        assert w.getsampwidth() == 2
        assert w.getnframes() / w.getframerate() == pytest.approx(1.0, abs=0.1)


def test_extract_audio_missing_video(tmp_path):
    with pytest.raises(ingest.AudioExtractionError, match="not found"):
        ingest.extract_audio(tmp_path / "nope.mp4", tmp_path / "audio.wav")


def test_extract_audio_ffmpeg_failure(tmp_path):
    bogus = tmp_path / "video.mp4"
    bogus.write_bytes(b"not a video")
    with pytest.raises(ingest.AudioExtractionError, match="ffmpeg failed"):
        ingest.extract_audio(bogus, tmp_path / "audio.wav")
    assert not (tmp_path / "audio.wav").exists()
    assert not (tmp_path / "audio.tmp.wav").exists()
