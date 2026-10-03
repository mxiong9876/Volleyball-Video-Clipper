import redis
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app import config
from app.db import get_engine
from app.routes import jobs, videos

app = FastAPI(title="Volley Breakdown API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(videos.router)
app.include_router(jobs.router)


def check_db() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def check_redis() -> bool:
    try:
        client = redis.Redis.from_url(
            config.REDIS_URL,
            socket_timeout=config.HEALTH_TIMEOUT,
            socket_connect_timeout=config.HEALTH_TIMEOUT,
        )
        return bool(client.ping())
    except Exception:
        return False


@app.get("/health")
def health(response: Response) -> dict:
    db_ok = check_db()
    redis_ok = check_redis()
    healthy = db_ok and redis_ok
    if not healthy:
        response.status_code = 503
    return {"status": "ok" if healthy else "degraded", "db": db_ok, "redis": redis_ok}
