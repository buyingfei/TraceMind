"""业务只读 SQL 统一超时与重试策略。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def sql_timeout_sec() -> int:
    """单次查询超时（秒），默认 300=5min。"""
    return max(1, int(get_settings().biz_readonly_timeout_sec))


def sql_max_attempts() -> int:
    """含首次的最大尝试次数 = 1 + retries。"""
    retries = max(0, int(get_settings().biz_readonly_sql_retries))
    return 1 + retries


def apply_session_timeout(conn: Connection, *, timeout_sec: int | None = None) -> None:
    """设置 MySQL 会话级 MAX_EXECUTION_TIME（毫秒）。"""
    sec = sql_timeout_sec() if timeout_sec is None else max(1, int(timeout_sec))
    try:
        conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={sec * 1000}"))
    except Exception:  # noqa: BLE001
        pass


def run_with_sql_retry(
    fn: Callable[[], T],
    *,
    label: str = "sql",
) -> T:
    """失败按配置重试；默认重试 3 次（共最多 4 次）。"""
    attempts = sql_max_attempts()
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "%s failed attempt=%s/%s: %s",
                label,
                i + 1,
                attempts,
                exc,
            )
            if i + 1 >= attempts:
                break
            # 短暂退避，避免打满只读库
            time.sleep(min(2**i, 8))
    assert last_exc is not None
    raise last_exc
