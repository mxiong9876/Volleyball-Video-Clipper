import pytest
from rq.exceptions import NoSuchJobError
from rq.job import JobStatus as RQStatus

from app import queue


class FakeJob:
    def __init__(self, job_id, status, worker_name=None):
        self.id, self.status, self.worker_name = job_id, status, worker_name

    def get_status(self, refresh=True):
        return self.status


class FakeRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, ex=None):
        self.store[key] = (value, ex)

    def exists(self, key):
        return int(key in self.store)


@pytest.fixture
def rq_state(monkeypatch):
    state = {"job": None, "redis": FakeRedis()}

    def fetch(job_id, connection=None):
        if state["job"] is None or state["job"].id != job_id:
            raise NoSuchJobError(job_id)
        return state["job"]

    monkeypatch.setattr(queue, "get_redis", lambda: state["redis"])
    monkeypatch.setattr(queue.Job, "fetch", staticmethod(fetch))
    return state


def test_enqueue_uses_deterministic_id():
    assert queue.rq_job_id(7) == "ingest-7"


def test_missing_rq_job_is_not_live(rq_state):
    assert queue.rq_job_is_live(7) is False


@pytest.mark.parametrize("status", [RQStatus.QUEUED, RQStatus.DEFERRED, RQStatus.SCHEDULED])
def test_waiting_rq_job_is_live(rq_state, status):
    rq_state["job"] = FakeJob("ingest-7", status)
    assert queue.rq_job_is_live(7) is True


def test_heartbeat_sets_expiring_key(rq_state):
    queue.send_heartbeat("w1")
    assert rq_state["redis"].store == {"volley:worker-alive:w1": (1, queue.HEARTBEAT_TTL)}


def test_started_job_with_live_worker_is_live(rq_state):
    rq_state["job"] = FakeJob("ingest-7", RQStatus.STARTED, worker_name="w1")
    queue.send_heartbeat("w1")
    assert queue.rq_job_is_live(7) is True


def test_started_job_whose_worker_died_is_not_live(rq_state):
    # Heartbeat key expired (or a different worker is the one beating).
    rq_state["job"] = FakeJob("ingest-7", RQStatus.STARTED, worker_name="w1")
    queue.send_heartbeat("w2")
    assert queue.rq_job_is_live(7) is False


def test_started_job_without_worker_name_is_not_live(rq_state):
    rq_state["job"] = FakeJob("ingest-7", RQStatus.STARTED)
    assert queue.rq_job_is_live(7) is False


@pytest.mark.parametrize(
    "status", [RQStatus.FINISHED, RQStatus.FAILED, RQStatus.STOPPED, RQStatus.CANCELED]
)
def test_ended_rq_job_is_not_live(rq_state, status):
    rq_state["job"] = FakeJob("ingest-7", status)
    assert queue.rq_job_is_live(7) is False
