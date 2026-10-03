import pytest
import redis
from sqlalchemy import func, select
from volley_pipeline import ingest

from app import queue
from app.models import AnalysisJob, JobStatus, Video
from tests.conftest import add_video

VID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VID}"


@pytest.fixture
def lookup(monkeypatch):
    """Replaces the YouTube lookup. Set .result or .error; .calls counts invocations."""

    class Lookup:
        result = {"youtube_id": VID, "title": "Finals", "duration_sec": 5400}
        error: Exception | None = None
        calls = 0

        def __call__(self, url, timeout=10):
            self.calls += 1
            if self.error:
                raise self.error
            return self.result

    fake = Lookup()
    monkeypatch.setattr(ingest, "fetch_metadata", fake)
    return fake


@pytest.fixture
def enqueued(monkeypatch):
    calls = []
    monkeypatch.setattr(queue, "enqueue_ingest", calls.append)
    return calls


@pytest.fixture
def live(monkeypatch):
    """Controls queue.rq_job_is_live; defaults to True (job still in Redis)."""
    state = {"live": True, "asked": []}

    def fake(job_id):
        state["asked"].append(job_id)
        return state["live"]

    monkeypatch.setattr(queue, "rq_job_is_live", fake)
    return state


def count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def test_submit_new_video(client, session, lookup, enqueued, live):
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 202
    body = resp.json()
    assert body["video"] == {"id": 1, "youtube_id": VID, "title": "Finals", "duration_sec": 5400}
    assert body["job"]["status"] == "queued"
    assert body["job"]["progress_pct"] == 0
    assert body["job"]["pipeline_version"]
    assert enqueued == [body["job"]["id"]]


def test_duplicate_returns_existing_without_lookup(client, session, lookup, enqueued, live):
    first = client.post("/videos", json={"url": URL}).json()
    resp = client.post("/videos", json={"url": f"https://youtu.be/{VID}?si=x"})
    assert resp.status_code == 200
    assert resp.json() == first
    assert lookup.calls == 1
    assert len(enqueued) == 1


def test_duplicate_of_done_video_does_not_check_queue(client, session, lookup, enqueued, live):
    add_video(session, status=JobStatus.DONE, progress_pct=100)
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 200
    assert resp.json()["job"]["status"] == "done"
    assert live["asked"] == []
    assert lookup.calls == 0


def test_duplicate_after_failure_starts_new_job(client, session, lookup, enqueued, live):
    add_video(session, status=JobStatus.FAILED, error_msg="Download failed: boom")
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 202
    job = resp.json()["job"]
    assert job["id"] == 2 and job["status"] == "queued"
    assert enqueued == [2]
    assert lookup.calls == 0


def test_duplicate_with_stale_job_retries(client, session, lookup, enqueued, live):
    add_video(session, status=JobStatus.DOWNLOADING, progress_pct=30)
    live["live"] = False
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 202
    assert resp.json()["job"]["id"] == 2
    assert enqueued == [2]
    session.expire_all()
    old = session.get(AnalysisJob, 1)
    assert old.status == JobStatus.FAILED
    assert old.error_msg == "worker stopped during processing"
    assert old.finished_at is not None


def test_duplicate_with_live_job_returns_it(client, session, lookup, enqueued, live):
    add_video(session, status=JobStatus.DOWNLOADING, progress_pct=30)
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 200
    assert resp.json()["job"]["id"] == 1
    assert resp.json()["job"]["status"] == "downloading"
    assert live["asked"] == [1]
    assert enqueued == []


@pytest.mark.parametrize("url", ["not a url", "https://vimeo.com/123", ""])
def test_invalid_url(client, session, lookup, enqueued, url):
    resp = client.post("/videos", json={"url": url})
    assert resp.status_code == 422
    assert "Not a valid YouTube video URL" in resp.json()["detail"]
    assert lookup.calls == 0


def test_too_long_rejected_without_row(client, session, lookup, enqueued):
    lookup.result = {"youtube_id": VID, "title": "Marathon", "duration_sec": 3 * 3600 + 1}
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Video is 3h 01m long; the limit is 3h"
    assert count(session, Video) == 0
    assert enqueued == []


def test_unavailable_video(client, session, lookup, enqueued):
    lookup.error = ingest.VideoUnavailableError("YouTube video unavailable: Private video")
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "YouTube video unavailable: Private video"
    assert count(session, Video) == 0


def test_lookup_timeout(client, session, lookup, enqueued):
    lookup.error = ingest.LookupTimeoutError("Couldn't reach YouTube to look up the video")
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 504
    assert "Couldn't reach YouTube" in resp.json()["detail"]
    assert count(session, Video) == 0


def test_lookup_hangs_past_deadline(client, session, lookup, enqueued, monkeypatch):
    import threading

    from app import config

    release = threading.Event()
    monkeypatch.setattr(config, "METADATA_TIMEOUT", 0.05)
    monkeypatch.setattr(ingest, "fetch_metadata", lambda url, timeout: release.wait(5))
    try:
        resp = client.post("/videos", json={"url": URL})
    finally:
        release.set()
    assert resp.status_code == 504
    assert "Timed out" in resp.json()["detail"]


def test_redis_down_on_enqueue(client, session, lookup, monkeypatch):
    def boom(job_id):
        raise redis.ConnectionError("refused")

    monkeypatch.setattr(queue, "enqueue_ingest", boom)
    resp = client.post("/videos", json={"url": URL})
    assert resp.status_code == 503
    session.expire_all()
    job = session.get(AnalysisJob, 1)
    assert job.status == JobStatus.FAILED
    assert "Redis" in job.error_msg
