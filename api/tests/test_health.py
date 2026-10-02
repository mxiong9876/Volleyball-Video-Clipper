import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def stub_checks(monkeypatch, db: bool, redis: bool) -> None:
    monkeypatch.setattr(main, "check_db", lambda: db)
    monkeypatch.setattr(main, "check_redis", lambda: redis)


def test_health_ok(monkeypatch):
    stub_checks(monkeypatch, db=True, redis=True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": True, "redis": True}


@pytest.mark.parametrize("db,redis", [(False, True), (True, False), (False, False)])
def test_health_degraded(monkeypatch, db, redis):
    stub_checks(monkeypatch, db=db, redis=redis)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "db": db, "redis": redis}
