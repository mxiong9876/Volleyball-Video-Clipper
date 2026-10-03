import os
from pathlib import Path

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://volley:volley@localhost:5432/volley"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

# Seconds each dependency check may take before /health reports it as down.
HEALTH_TIMEOUT = 2

# Raw media lives at <repo>/data/raw/<youtube_id>/ (absolute, so it works from any cwd).
DATA_DIR = Path(
    os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "raw")
).resolve()

# Ingestion limits.
MAX_DURATION_SEC = 3 * 3600
# Seconds POST /videos waits for the YouTube metadata lookup before giving up.
METADATA_TIMEOUT = 10

QUEUE_NAME = "ingest"
# Upper bound for one ingest job (a 3h video at 720p on a slow connection).
INGEST_JOB_TIMEOUT = 2 * 3600
