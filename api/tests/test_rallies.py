from datetime import timedelta

import pytest
from sqlalchemy import select
from volley_pipeline import detectors, ingest, motion

from app import config, tasks
from app.models import AnalysisJob, JobStatus, Rally, utcnow
from app.rallies import replace_rallies
from tests.conftest import add_video


@pytest.fixture
def fake_ingest(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        ingest, "download_video", lambda url, out_dir, on_progress=None: out_dir / "video.mp4"
    )
    monkeypatch.setattr(ingest, "extract_audio", lambda video, out: out)


def stored(session_factory, source=None) -> list[tuple]:
    with session_factory() as s:
        query = select(Rally).order_by(Rally.source, Rally.rally_index)
        if source:
            query = query.where(Rally.source == source)
        return [(r.source, r.rally_index, r.start_sec, r.end_sec) for r in s.scalars(query)]


def load_job(session_factory) -> AnalysisJob:
    with session_factory() as s:
        return s.get(AnalysisJob, 1)


def test_ingest_runs_current_detector_and_stores_rallies(
    session, session_factory, fake_ingest, fake_detector, tmp_path
):
    video = add_video(session, status=JobStatus.QUEUED)
    during = []

    def snapshot():
        job = load_job(session_factory)
        during.append((job.status, job.progress_pct, stored(session_factory)))

    fake_detector.on_call = snapshot
    tasks.ingest_video(1)

    assert fake_detector.calls == [(detectors.CURRENT, tmp_path / video.youtube_id)]
    assert during == [(JobStatus.DETECTING_RALLIES, 90, [])]
    assert stored(session_factory) == [
        (detectors.CURRENT, 0, 1.0, 5.0),
        (detectors.CURRENT, 1, 10.12, 20.46),  # rounded like the prediction files
    ]
    job = load_job(session_factory)
    assert (job.status, job.progress_pct) == (JobStatus.DONE, 100)


def test_zero_rallies_is_still_done(session, session_factory, fake_ingest, fake_detector):
    add_video(session, status=JobStatus.QUEUED)
    fake_detector.rallies = []
    tasks.ingest_video(1)
    assert load_job(session_factory).status == JobStatus.DONE
    assert stored(session_factory) == []


def test_detector_failure_fails_job_and_keeps_old_rows(
    session, session_factory, fake_ingest, fake_detector
):
    video = add_video(session, status=JobStatus.QUEUED)
    replace_rallies(session, video.id, detectors.CURRENT, [(1.0, 2.0)])
    session.commit()
    fake_detector.error = motion.MotionError("ffmpeg failed to read frames: bad file")

    with pytest.raises(motion.MotionError):
        tasks.ingest_video(1)

    job = load_job(session_factory)
    assert job.status == JobStatus.FAILED
    assert job.error_msg == "ffmpeg failed to read frames: bad file"
    assert stored(session_factory) == [(detectors.CURRENT, 0, 1.0, 2.0)]


def test_failure_while_storing_leaves_nothing_partial(
    session, session_factory, fake_ingest, fake_detector
):
    video = add_video(session, status=JobStatus.QUEUED)
    replace_rallies(session, video.id, detectors.CURRENT, [(1.0, 2.0)])
    session.commit()
    fake_detector.rallies = [(3.0, 4.0), (None, 5.0)]  # sorting these raises mid-store

    with pytest.raises(TypeError):
        tasks.ingest_video(1)

    assert load_job(session_factory).status == JobStatus.FAILED
    assert stored(session_factory) == [(detectors.CURRENT, 0, 1.0, 2.0)]


def test_fail_stale_jobs_covers_detecting_rallies(session, session_factory):
    add_video(session, status=JobStatus.DETECTING_RALLIES)
    assert tasks.fail_stale_jobs(session, is_live=lambda job_id: True) == 1
    assert load_job(session_factory).status == JobStatus.FAILED


def test_replace_rallies_only_touches_its_source(session, session_factory):
    video = add_video(session)
    replace_rallies(session, video.id, "label", [(1.0, 2.0)])
    replace_rallies(session, video.id, "motion_v1", [(5.0, 6.0), (3.0, 4.0)])
    session.commit()
    replace_rallies(session, video.id, "motion_v1", [(7.0, 8.0)])
    session.commit()
    assert stored(session_factory) == [("label", 0, 1.0, 2.0), ("motion_v1", 0, 7.0, 8.0)]


def test_replace_rallies_indexes_in_time_order(session, session_factory):
    video = add_video(session)
    assert replace_rallies(session, video.id, "x", [(5.0, 6.0), (3.0, 4.0)]) == 2
    session.commit()
    assert stored(session_factory) == [("x", 0, 3.0, 4.0), ("x", 1, 5.0, 6.0)]


def test_list_videos_empty(client):
    response = client.get("/videos")
    assert response.status_code == 200
    assert response.json() == []


def test_list_videos_newest_first_with_latest_job(client, session):
    old = add_video(session, youtube_id="video000000", status=JobStatus.FAILED)
    old.created_at = utcnow() - timedelta(days=1)
    session.add(AnalysisJob(video=old, pipeline_version="0.0.0", status=JobStatus.DONE))
    add_video(session, youtube_id="video000001")  # no job yet
    session.commit()

    body = client.get("/videos").json()

    assert [v["youtube_id"] for v in body] == ["video000001", "video000000"]
    assert body[0]["latest_job"] is None
    assert body[1]["latest_job"]["status"] == JobStatus.DONE
    assert body[1]["title"] == "Match" and body[1]["duration_sec"] == 3600


def test_get_rallies_unknown_video(client):
    response = client.get("/videos/42/rallies")
    assert response.status_code == 404
    assert response.json()["detail"] == "Video 42 not found"


def test_get_rallies_all_sources_and_filter(client, session):
    video = add_video(session)
    other = add_video(session, youtube_id="video000001")
    replace_rallies(session, video.id, detectors.CURRENT, [(1.0, 2.0), (3.0, 4.0)])
    replace_rallies(session, video.id, "label", [(1.5, 2.5)])
    replace_rallies(session, other.id, "label", [(9.0, 10.0)])
    session.commit()

    body = client.get(f"/videos/{video.id}/rallies").json()
    assert body["video_id"] == video.id
    assert body["detector"] == detectors.CURRENT
    rows = [(r["source"], r["rally_index"], r["start_sec"], r["end_sec"]) for r in body["rallies"]]
    assert rows == sorted(
        [
            (detectors.CURRENT, 0, 1.0, 2.0),
            (detectors.CURRENT, 1, 3.0, 4.0),
            ("label", 0, 1.5, 2.5),
        ]
    )
    assert body["rallies"][0]["created_at"]

    labels = client.get(f"/videos/{video.id}/rallies", params={"source": "label"}).json()
    assert [r["source"] for r in labels["rallies"]] == ["label"]


def test_get_rallies_none_yet(client, session):
    video = add_video(session)
    assert client.get(f"/videos/{video.id}/rallies").json()["rallies"] == []
