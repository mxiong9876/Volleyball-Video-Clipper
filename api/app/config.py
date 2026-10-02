import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://volley:volley@localhost:5432/volley"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

# Seconds each dependency check may take before /health reports it as down.
HEALTH_TIMEOUT = 2
