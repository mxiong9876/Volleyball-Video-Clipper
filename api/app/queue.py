from functools import lru_cache

import redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job
from rq.job import JobStatus as RQStatus

from app import config

_WAITING = {RQStatus.QUEUED, RQStatus.DEFERRED, RQStatus.SCHEDULED}

# worker.py refreshes a short-lived key while it's alive. RQ's own worker registration
# can't be used for this: it's kept for the whole job timeout (hours) after a hard kill.
HEARTBEAT_TTL = 30
HEARTBEAT_INTERVAL = 10


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(config.REDIS_URL)


def get_queue() -> Queue:
    return Queue(config.QUEUE_NAME, connection=get_redis())


def rq_job_id(job_id: int) -> str:
    """Deterministic RQ id, so an analysis_jobs row can be matched to its RQ job."""
    return f"ingest-{job_id}"


def enqueue_ingest(job_id: int) -> None:
    get_queue().enqueue_call(
        "app.tasks.ingest_video",
        args=(job_id,),
        job_id=rq_job_id(job_id),
        timeout=config.INGEST_JOB_TIMEOUT,
    )


def heartbeat_key(worker_name: str) -> str:
    return f"volley:worker-alive:{worker_name}"


def send_heartbeat(worker_name: str) -> None:
    get_redis().set(heartbeat_key(worker_name), 1, ex=HEARTBEAT_TTL)


def rq_job_is_live(job_id: int) -> bool:
    """True if the RQ job is waiting to run, or running on a worker that's still alive.

    A worker killed mid-job leaves the RQ job "started"; once its heartbeat key
    expires (HEARTBEAT_TTL) that counts as not live. Raises redis.RedisError if
    Redis is unreachable.
    """
    conn = get_redis()
    try:
        job = Job.fetch(rq_job_id(job_id), connection=conn)
    except NoSuchJobError:
        return False
    status = job.get_status(refresh=False)
    if status in _WAITING:
        return True
    if status == RQStatus.STARTED:
        return bool(job.worker_name) and bool(conn.exists(heartbeat_key(job.worker_name)))
    return False
