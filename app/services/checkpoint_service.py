"""LangGraph Postgres Checkpointer（Agent 短期记忆）。"""

from __future__ import annotations

import logging
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_checkpointer: Any = None
_setup_done = False


def _conninfo() -> str:
    return (get_settings().langgraph_database_url or "").strip()


def get_checkpoint_pool() -> ConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    conninfo = _conninfo()
    if not conninfo:
        raise RuntimeError("未配置 LANGGRAPH_DATABASE_URL")
    _pool = ConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=8,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=True,
    )
    logger.info("LangGraph checkpoint pool ready")
    return _pool


def get_checkpointer():
    """返回进程内单例 PostgresSaver。"""
    global _checkpointer, _setup_done
    if _checkpointer is not None:
        return _checkpointer

    from langgraph.checkpoint.postgres import PostgresSaver

    pool = get_checkpoint_pool()
    saver = PostgresSaver(pool)
    if not _setup_done:
        saver.setup()
        _setup_done = True
        logger.info("LangGraph checkpoint tables ensured")
    _checkpointer = saver
    return _checkpointer


def thread_config(conversation_id: int) -> dict[str, Any]:
    """conversation_id ↔ LangGraph thread_id。"""
    return {"configurable": {"thread_id": str(conversation_id)}}


def ensure_checkpoint_ready() -> dict[str, str]:
    """应用启动时调用：建表并探测连通性。"""
    get_checkpointer()
    return {"status": "ok", "database": _conninfo()}
