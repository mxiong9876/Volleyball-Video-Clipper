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
