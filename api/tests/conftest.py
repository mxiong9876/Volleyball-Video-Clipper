import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import db, models
from app.main import app


@pytest.fixture
def session_factory(monkeypatch):
    """In-memory SQLite shared by the API (get_session) and tasks (db.SessionLocal)."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    db.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db, "SessionLocal", factory)

    def override():
        with factory() as session:
            yield session

    app.dependency_overrides[db.get_session] = override
    yield factory
    app.dependency_overrides.pop(db.get_session, None)
    engine.dispose()


@pytest.fixture
def session(session_factory) -> Session:
    with session_factory() as s:
        yield s


@pytest.fixture
def client(session_factory) -> TestClient:
    return TestClient(app)


def add_video(session: Session, youtube_id: str = "dQw4w9WgXcQ", **job_fields) -> models.Video:
    """Insert a video, plus one job if job_fields are given."""
    video = models.Video(youtube_id=youtube_id, title="Match", duration_sec=3600)
    session.add(video)
    if job_fields:
        session.add(models.AnalysisJob(video=video, pipeline_version="0.0.0", **job_fields))
    session.commit()
    return video
