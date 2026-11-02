import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from volley_pipeline import detectors

from app import db, models
from app.main import app


class FakeDetector:
    """Stands in for detectors.detect. Set .rallies, .error or .on_call; .calls records uses."""

    def __init__(self):
        self.rallies = [(1.0, 5.0), (10.123, 20.456)]
        self.error = None
        self.on_call = None
        self.calls = []

    def __call__(self, name, media_dir):
        self.calls.append((name, media_dir))
        if self.on_call:
            self.on_call()
        if self.error:
            raise self.error
        return self.rallies


@pytest.fixture(autouse=True)
def fake_detector(monkeypatch) -> FakeDetector:
    """API tests never decode real media; the detectors have their own tests in pipeline/."""
    fake = FakeDetector()
    monkeypatch.setattr(detectors, "detect", fake)
    return fake


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
