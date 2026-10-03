from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    DONE = "done"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset({JobStatus.DONE, JobStatus.FAILED})


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    youtube_id: Mapped[str] = mapped_column(String(11), unique=True)
    title: Mapped[str] = mapped_column(Text)
    duration_sec: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    jobs: Mapped[list["AnalysisJob"]] = relationship(
        back_populates="video", order_by="AnalysisJob.id"
    )


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    # VARCHAR + CHECK instead of a native PG enum so later phases can add statuses cheaply.
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'downloading', 'extracting_audio', 'done', 'failed')",
            name="ck_analysis_jobs_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.QUEUED)
    progress_pct: Mapped[int] = mapped_column(default=0)
    pipeline_version: Mapped[str] = mapped_column(String(32))
    error_msg: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    video: Mapped[Video] = relationship(back_populates="jobs")
