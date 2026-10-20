"""SQLAlchemy MySQL 连接（仅引擎/会话，不自动建表）。"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def get_session() -> Generator[Session, None, None]:
    """获取会话（用于依赖注入）；用完自动关闭。"""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def ping_database() -> dict:
    """执行 SELECT 1，验证连接是否可用（不建表）。"""
    settings = get_settings()
    with get_engine().connect() as conn:
        value = conn.execute(text("SELECT 1")).scalar()
    return {
        "ok": value == 1,
        "host": settings.mysql_host,
        "port": settings.mysql_port,
        "database": settings.mysql_database,
        "driver": "mysql+pymysql",
    }
