from app.models import JobStatus
from tests.conftest import add_video


def test_get_job(client, session):
    add_video(session, status=JobStatus.DOWNLOADING, progress_pct=42)
    resp = client.get("/jobs/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == 1
    assert body["video_id"] == 1
    assert body["status"] == "downloading"
    assert body["progress_pct"] == 42
    assert body["error_msg"] is None
    assert body["pipeline_version"] == "0.0.0"
    assert set(body) == {
        "id", "video_id", "status", "progress_pct", "error_msg", "pipeline_version",
        "created_at", "started_at", "finished_at",
    }  # fmt: skip


def test_get_job_not_found(client, session):
    resp = client.get("/jobs/999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job 999 not found"
