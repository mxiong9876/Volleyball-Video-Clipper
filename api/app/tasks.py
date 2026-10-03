"""RQ tasks. They call volley_pipeline functions and record progress in the DB."""

import logging
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session
from volley_pipeline import ingest

from app import config, db, queue
from app.models import TERMINAL_STATUSES, AnalysisJob, JobStatus, utcnow

log = logging.getLogger(__name__)

WORKER_STOPPED_MSG = "worker stopped during processing"
# Share of progress_pct given to the download; audio extraction covers the rest.
DOWNLOAD_PCT = 80
EXTRACTING_PCT = 85


def _update(session: Session, job: AnalysisJob, **fields) -> None:
    for name, value in fields.items():
        setattr(job, name, value)
    session.commit()


def mark_failed(session: Session, job: AnalysisJob, msg: str) -> None:
    _update(session, job, status=JobStatus.FAILED, error_msg=msg, finished_at=utcnow())


def ingest_video(job_id: int) -> None:
    with db.SessionLocal() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            log.warning("analysis job %s not found; skipping", job_id)
            return
        if job.status in TERMINAL_STATUSES:
            log.info("analysis job %s already %s; skipping", job_id, job.status)
            return

        youtube_id = job.video.youtube_id
        out_dir = config.DATA_DIR / youtube_id
        url = f"https://www.youtube.com/watch?v={youtube_id}"

        def on_progress(pct: float) -> None:
            value = int(pct * DOWNLOAD_PCT / 100)
            if value != job.progress_pct:  # only write when the whole percent changes
                _update(session, job, progress_pct=value)

        try:
            _update(session, job, status=JobStatus.DOWNLOADING, started_at=utcnow(), progress_pct=0)
            video_path = ingest.download_video(url, out_dir, on_progress=on_progress)
            _update(session, job, status=JobStatus.EXTRACTING_AUDIO, progress_pct=EXTRACTING_PCT)
            ingest.extract_audio(video_path, out_dir / ingest.AUDIO_FILENAME)
            _update(session, job, status=JobStatus.DONE, progress_pct=100, finished_at=utcnow())
        except Exception as exc:
            session.rollback()
            if isinstance(exc, ingest.IngestError):
                msg = str(exc)
            else:
                msg = f"Internal error: {type(exc).__name__}: {exc}"
            mark_failed(session, job, msg)
            raise


def fail_stale_jobs(session: Session, is_live: Callable[[int], bool] = queue.rq_job_is_live) -> int:
    """Run at worker startup: fail jobs a previous worker left unfinished.

    Assumes a single worker: anything mid-processing when this worker starts was
    orphaned by one that died. Queued jobs still waiting in Redis are left alone.
    Returns how many jobs were failed.
    """
    jobs = session.scalars(
        select(AnalysisJob).where(AnalysisJob.status.not_in(list(TERMINAL_STATUSES)))
    ).all()
    failed = 0
    for job in jobs:
        if job.status == JobStatus.QUEUED and is_live(job.id):
            continue
        mark_failed(session, job, WORKER_STOPPED_MSG)
        failed += 1
    return failed
