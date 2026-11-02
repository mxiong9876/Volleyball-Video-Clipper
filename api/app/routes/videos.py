from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime

import redis
import volley_pipeline
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from volley_pipeline import detectors, ingest

from app import config, queue, tasks
from app.db import SessionDep
from app.models import AnalysisJob, JobStatus, Rally, Video
from app.routes.jobs import JobOut

router = APIRouter()

# The yt-dlp lookup runs here so the request can give up after METADATA_TIMEOUT even
# if yt-dlp's per-socket timeout keeps it going longer.
_lookup_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yt-lookup")

QUEUE_DOWN_MSG = "Job queue unavailable (Redis unreachable); try again shortly"


class SubmitVideo(BaseModel):
    url: str


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    youtube_id: str
    title: str
    duration_sec: int


class SubmitResult(BaseModel):
    video: VideoOut
    job: JobOut


class VideoListItem(VideoOut):
    latest_job: JobOut | None


class RallyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rally_index: int
    start_sec: float
    end_sec: float
    source: str
    created_at: datetime


class VideoRallies(BaseModel):
    video_id: int
    detector: str  # the source whose rows are the live predictions; others are labels or older runs
    rallies: list[RallyOut]


def _result(video: Video, job: AnalysisJob) -> SubmitResult:
    return SubmitResult(video=VideoOut.model_validate(video), job=JobOut.model_validate(job))


def _lookup_metadata(url: str) -> dict:
    future = _lookup_pool.submit(ingest.fetch_metadata, url, config.METADATA_TIMEOUT)
    try:
        return future.result(timeout=config.METADATA_TIMEOUT)
    except (FutureTimeout, ingest.LookupTimeoutError) as exc:
        detail = "Timed out looking up the video on YouTube; try again shortly"
        if isinstance(exc, ingest.LookupTimeoutError):
            detail = str(exc)
        raise HTTPException(status_code=504, detail=detail) from exc
    except (ingest.VideoUnavailableError, ingest.InvalidURLError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _start_job(session: Session, video: Video) -> AnalysisJob:
    """Create a queued job for video, commit it, and enqueue it on RQ."""
    job = AnalysisJob(
        video=video, status=JobStatus.QUEUED, pipeline_version=volley_pipeline.__version__
    )
    session.add(job)
    session.commit()
    try:
        queue.enqueue_ingest(job.id)
    except redis.RedisError as exc:
        tasks.mark_failed(session, job, QUEUE_DOWN_MSG)
        raise HTTPException(status_code=503, detail=QUEUE_DOWN_MSG) from exc
    return job


def _existing(session: Session, video: Video, response: Response) -> SubmitResult:
    """Return the video's current job, or start a new one if the last one can't finish."""
    latest = video.jobs[-1] if video.jobs else None
    if latest is not None and latest.status != JobStatus.FAILED:
        try:
            alive = latest.status == JobStatus.DONE or queue.rq_job_is_live(latest.id)
        except redis.RedisError as exc:
            raise HTTPException(status_code=503, detail=QUEUE_DOWN_MSG) from exc
        if alive:
            response.status_code = 200
            return _result(video, latest)
        tasks.mark_failed(session, latest, tasks.WORKER_STOPPED_MSG)
    job = _start_job(session, video)
    session.refresh(video)
    return _result(video, job)


@router.post(
    "/videos",
    response_model=SubmitResult,
    status_code=202,
    responses={
        200: {"description": "Already submitted; returns the existing video and job"},
        422: {"description": "Invalid URL, unavailable video, or longer than the limit"},
        503: {"description": "Job queue unavailable"},
        504: {"description": "YouTube lookup timed out"},
    },
)
def submit_video(body: SubmitVideo, response: Response, session: SessionDep) -> SubmitResult:
    try:
        youtube_id = ingest.parse_youtube_id(body.url)
    except ingest.InvalidURLError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Duplicates are answered from the DB, before any network call.
    video = session.scalar(select(Video).where(Video.youtube_id == youtube_id))
    if video is not None:
        return _existing(session, video, response)

    meta = _lookup_metadata(body.url)
    try:
        ingest.check_duration(meta["duration_sec"], config.MAX_DURATION_SEC)
    except ingest.VideoTooLongError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    video = Video(youtube_id=youtube_id, title=meta["title"], duration_sec=meta["duration_sec"])
    session.add(video)
    try:
        session.flush()
    except IntegrityError:
        # A concurrent request inserted the same video first; use that one.
        session.rollback()
        video = session.scalar(select(Video).where(Video.youtube_id == youtube_id))
        return _existing(session, video, response)
    return _result(video, _start_job(session, video))


@router.get("/videos", response_model=list[VideoListItem])
def list_videos(session: SessionDep) -> list[VideoListItem]:
    videos = session.scalars(
        select(Video)
        .options(selectinload(Video.jobs))
        .order_by(Video.created_at.desc(), Video.id.desc())
    ).all()
    return [
        VideoListItem(
            **VideoOut.model_validate(video).model_dump(),
            latest_job=JobOut.model_validate(video.jobs[-1]) if video.jobs else None,
        )
        for video in videos
    ]


@router.get("/videos/{video_id}/rallies", response_model=VideoRallies)
def get_rallies(video_id: int, session: SessionDep, source: str | None = None) -> VideoRallies:
    if session.get(Video, video_id) is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    query = select(Rally).where(Rally.video_id == video_id)
    if source is not None:
        query = query.where(Rally.source == source)
    rallies = session.scalars(query.order_by(Rally.source, Rally.rally_index)).all()
    return VideoRallies(
        video_id=video_id,
        detector=detectors.CURRENT,
        rallies=[RallyOut.model_validate(r) for r in rallies],
    )
