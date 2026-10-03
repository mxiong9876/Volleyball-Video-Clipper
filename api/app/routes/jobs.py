from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from app.db import SessionDep
from app.models import AnalysisJob

router = APIRouter()


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    video_id: int
    status: str
    progress_pct: int
    error_msg: str | None
    pipeline_version: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, session: SessionDep) -> AnalysisJob:
    job = session.get(AnalysisJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job
