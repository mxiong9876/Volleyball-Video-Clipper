"""RQ worker for ingest jobs: `make worker` (or `uv run python worker.py` from api/)."""

import logging
import threading

import redis
from rq import SimpleWorker

from app import db, queue, tasks

log = logging.getLogger("worker")


def _heartbeat_loop(worker_name: str, stop: threading.Event) -> None:
    """Keeps this worker's liveness key fresh so POST /videos can spot a dead worker."""
    while not stop.wait(queue.HEARTBEAT_INTERVAL):
        try:
            queue.send_heartbeat(worker_name)
        except redis.RedisError as exc:
            log.warning("heartbeat failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with db.SessionLocal() as session:
        failed = tasks.fail_stale_jobs(session)
    if failed:
        log.info("marked %d unfinished job(s) from a previous worker as failed", failed)

    # SimpleWorker runs jobs in-process (no fork), avoiding macOS fork-safety crashes.
    worker = SimpleWorker([queue.get_queue()], connection=queue.get_redis())
    queue.send_heartbeat(worker.name)
    stop = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(worker.name, stop), daemon=True).start()
    try:
        worker.work()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
