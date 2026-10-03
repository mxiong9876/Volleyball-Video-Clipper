from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app import config


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine() -> Engine:
    return create_engine(
        config.DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"connect_timeout": config.HEALTH_TIMEOUT},
    )


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def SessionLocal() -> Session:
    return _session_factory()()


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    with SessionLocal() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
