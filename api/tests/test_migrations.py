"""Runs the Alembic migrations against a real Postgres (the docker one from `make up`).

Uses a separate `volley_test` database so downgrading never touches dev data.
Skipped when Postgres is unreachable so `make test` works without docker.
"""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from app import config

TEST_DB = "volley_test"
ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


@pytest.fixture(scope="module")
def test_db_url() -> str:
    url = make_url(os.environ.get("TEST_DATABASE_URL", config.DATABASE_URL))
    admin = create_engine(url, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 2})
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": TEST_DB}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{TEST_DB}"'))
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({type(exc).__name__}); run `make up`")
    finally:
        admin.dispose()
    return url.set(database=TEST_DB).render_as_string(hide_password=False)


def alembic_config(url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def test_upgrade_and_downgrade(test_db_url):
    cfg = alembic_config(test_db_url)
    engine = create_engine(test_db_url)
    try:
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")

        insp = inspect(engine)
        assert {"videos", "analysis_jobs"} <= set(insp.get_table_names())
        uniques = insp.get_unique_constraints("videos")
        assert any(u["column_names"] == ["youtube_id"] for u in uniques)
        checks = insp.get_check_constraints("analysis_jobs")
        assert any(c["name"] == "ck_analysis_jobs_status" for c in checks)

        command.downgrade(cfg, "base")
        assert not {"videos", "analysis_jobs"} & set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_rallies_table_and_detecting_rallies_status(test_db_url):
    cfg = alembic_config(test_db_url)
    engine = create_engine(test_db_url)
    insert_video = text(
        "INSERT INTO videos (youtube_id, title, duration_sec, created_at) "
        "VALUES ('dQw4w9WgXcQ', 'Match', 60, now()) RETURNING id"
    )
    insert_job = text(
        "INSERT INTO analysis_jobs (video_id, status, progress_pct, pipeline_version, created_at) "
        "VALUES (:video_id, 'detecting_rallies', 90, '0.0.0', now())"
    )
    try:
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")

        insp = inspect(engine)
        assert "rallies" in insp.get_table_names()
        assert any(
            u["column_names"] == ["video_id", "source", "rally_index"]
            for u in insp.get_unique_constraints("rallies")
        )
        with engine.begin() as conn:
            video_id = conn.execute(insert_video).scalar_one()
            conn.execute(insert_job, {"video_id": video_id})

        # A job caught mid-stage must not block the downgrade.
        command.downgrade(cfg, "0001")
        insp = inspect(engine)
        assert "rallies" not in insp.get_table_names()
        with engine.connect() as conn:
            status = conn.execute(text("SELECT status FROM analysis_jobs")).scalar_one()
        assert status == "failed"
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(insert_job, {"video_id": video_id})
    finally:
        command.downgrade(cfg, "base")
        engine.dispose()
