"""业务只读库（与本系统 MySQL 分离）。"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import get_settings


@lru_cache
def get_biz_readonly_engine() -> Engine:
    cfg = get_settings()
    url = (cfg.biz_readonly_database_url or "").strip()
    if not url:
        raise RuntimeError("未配置 BIZ_READONLY_DATABASE_URL（业务只读库）")
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
        pool_recycle=1800,
    )


def biz_readonly_enabled() -> bool:
    cfg = get_settings()
    return bool((cfg.biz_readonly_database_url or "").strip()) and cfg.biz_readonly_sql_enabled


def dispose_biz_readonly_engine() -> None:
    get_biz_readonly_engine.cache_clear()
