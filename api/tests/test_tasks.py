import pytest
from volley_pipeline import ingest

from app import config, tasks
from app.models import AnalysisJob, JobStatus
from tests.conftest import add_video

VID = "dQw4w9WgXcQ"


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path, session_factory):
    """Fakes download/extract; records the job's DB state at each pipeline call."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    seen = {"states": [], "download": None, "extract": None, "fail": None}

    def snapshot(job_id):
        with session_factory() as s:
            job = s.get(AnalysisJob, job_id)
            seen["states"].append((job.status, job.progress_pct))

    def download(url, out_dir, on_progress=None):
        seen["download"] = (url, out_dir)
        snapshot(1)
        for pct in (10.0, 10.4, 50.0, 100.0):
            on_progress(pct)
        snapshot(1)
        if seen["fail"] == "download":
            raise ingest.DownloadError("Download failed: HTTP Error 403")
        return out_dir / "video.mp4"

    def extract(video_path, out_path):
        seen["extract"] = (video_path, out_path)
        snapshot(1)
        if seen["fail"] == "extract":
            raise RuntimeError("disk full")
        return out_path

    monkeypatch.setattr(ingest, "download_video", download)
    monkeypatch.setattr(ingest, "extract_audio", extract)
    return seen


def load_job(session_factory, job_id=1) -> AnalysisJob:
    with session_factory() as s:
        return s.get(AnalysisJob, job_id)


def test_ingest_video_happy_path(session, session_factory, fake_pipeline, tmp_path):
    add_video(session, status=JobStatus.QUEUED)
    tasks.ingest_video(1)

    assert fake_pipeline["states"] == [
        (JobStatus.DOWNLOADING, 0),
        (JobStatus.DOWNLOADING, 80),  # 100% of the download share
        (JobStatus.EXTRACTING_AUDIO, 85),
    ]
    assert fake_pipeline["download"] == (f"https://www.youtube.com/watch?v={VID}", tmp_path / VID)
    assert fake_pipeline["extract"] == (tmp_path / VID / "video.mp4", tmp_path / VID / "audio.wav")
    job = load_job(session_factory)
    assert job.status == JobStatus.DONE
    assert job.progress_pct == 100
    assert job.started_at is not None and job.finished_at is not None
    assert job.error_msg is None


def test_ingest_video_ingest_error(session, session_factory, fake_pipeline):
    add_video(session, status=JobStatus.QUEUED)
    fake_pipeline["fail"] = "download"
    with pytest.raises(ingest.DownloadError):
        tasks.ingest_video(1)
    job = load_job(session_factory)
    assert job.status == JobStatus.FAILED
    assert job.error_msg == "Download failed: HTTP Error 403"
    assert job.finished_at is not None


def test_ingest_video_unexpected_error(session, session_factory, fake_pipeline):
    add_video(session, status=JobStatus.QUEUED)
    fake_pipeline["fail"] = "extract"
    with pytest.raises(RuntimeError):
        tasks.ingest_video(1)
    job = load_job(session_factory)
    assert job.status == JobStatus.FAILED
    assert job.error_msg == "Internal error: RuntimeError: disk full"


@pytest.mark.parametrize("status", [JobStatus.DONE, JobStatus.FAILED])
def test_ingest_video_skips_terminal_job(session, session_factory, fake_pipeline, status):
    add_video(session, status=status)
    tasks.ingest_video(1)
    assert fake_pipeline["download"] is None
    assert load_job(session_factory).status == status


def test_ingest_video_missing_job(session_factory, fake_pipeline):
    tasks.ingest_video(42)  # no exception
    assert fake_pipeline["download"] is None


def test_fail_stale_jobs(session, session_factory):
    rows = {
        "queued_live": JobStatus.QUEUED,
        "queued_gone": JobStatus.QUEUED,
        "downloading": JobStatus.DOWNLOADING,
        "extracting": JobStatus.EXTRACTING_AUDIO,
        "done": JobStatus.DONE,
        "failed": JobStatus.FAILED,
    }
    ids = {}
    for i, (name, status) in enumerate(rows.items()):
        video = add_video(session, youtube_id=f"video{i:06d}", status=status)
        ids[name] = video.jobs[0].id
    live_ids = {ids["queued_live"], ids["downloading"]}  # "downloading" is failed regardless

    assert tasks.fail_stale_jobs(session, is_live=lambda job_id: job_id in live_ids) == 3

    def state(name):
        job = load_job(session_factory, ids[name])
        return job.status, job.error_msg

    stopped = (JobStatus.FAILED, "worker stopped during processing")
    assert state("queued_live") == (JobStatus.QUEUED, None)
    assert state("queued_gone") == stopped
    assert state("downloading") == stopped
    assert state("extracting") == stopped
    assert state("done") == (JobStatus.DONE, None)
    assert state("failed") == (JobStatus.FAILED, None)
