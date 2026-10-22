"""业务库只读 SQL（SELECT/WITH only）。

与 app/db（本系统 MySQL）分离；Engine 工厂见 app.infra.biz_db。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import text

from app.core.config import get_settings
from app.infra.biz_db import biz_readonly_enabled, get_biz_readonly_engine

# 兼容旧 import 路径（工具 / 节点仍可从本模块取 Engine 开关）
__all__ = [
    "READONLY_SQL_TOOL",
    "biz_readonly_enabled",
    "get_biz_readonly_engine",
    "run_select",
]

logger = logging.getLogger(__name__)

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|"
    r"CALL|EXEC|EXECUTE|LOAD|OUTFILE|DUMPFILE|INTO\s+OUTFILE|LOCK|UNLOCK|SET)\b",
    re.IGNORECASE,
)
_ALLOWED_START = re.compile(r"^\s*(WITH|SELECT)\b", re.IGNORECASE | re.DOTALL)


def _normalize_sql(sql: str) -> str:
    raw = (sql or "").strip()
    if not raw:
        raise ValueError("SQL 为空")
    # 去掉块注释与行注释（简单处理）
    raw = re.sub(r"/\*.*?\*/", " ", raw, flags=re.DOTALL)
    raw = re.sub(r"--[^\n]*", " ", raw)
    raw = raw.strip().rstrip(";").strip()
    if ";" in raw:
        raise ValueError("仅允许单条 SQL 语句")
    if not _ALLOWED_START.search(raw):
        raise ValueError("仅允许 SELECT / WITH ... SELECT 查询")
    if _FORBIDDEN.search(raw):
        raise ValueError("检测到非只读关键字，已拒绝执行")
    return raw


def run_select(
    sql: str,
    params: dict | None = None,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """执行只读查询，返回 rows / columns / truncated。"""
    if not biz_readonly_enabled():
        raise RuntimeError("业务只读 SQL 未启用或未配置连接串")

    cfg = get_settings()
    limit = max_rows if max_rows is not None else int(cfg.biz_readonly_max_rows)
    limit = max(1, min(limit, 200))
    safe_sql = _normalize_sql(sql)
    # 若模型已写 LIMIT，去掉末尾 LIMIT 再统一包一层，避免双重 LIMIT
    safe_sql = re.sub(
        r"\s+LIMIT\s+\d+(\s*,\s*\d+)?\s*$",
        "",
        safe_sql,
        flags=re.IGNORECASE,
    )

    # 外层再包一层 LIMIT，防止模型漏写
    wrapped = f"SELECT * FROM ({safe_sql}) AS _diag_ro LIMIT {limit + 1}"

    engine = get_biz_readonly_engine()
    timeout = max(1, int(cfg.biz_readonly_timeout_sec))
    try:
        with engine.connect() as conn:
            try:
                conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={timeout * 1000}"))
            except Exception:  # noqa: BLE001
                pass
            try:
                result = conn.execute(text(wrapped), params or {})
            except Exception:
                # 某些语句不便包一层子查询时，退回原 SQL + LIMIT
                fallback_sql = f"{safe_sql} LIMIT {limit + 1}"
                result = conn.execute(text(fallback_sql), params or {})
            columns = list(result.keys())
            fetched = result.fetchmany(limit + 1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("只读 SQL 执行失败: %s | sql=%s", exc, safe_sql[:200])
        raise RuntimeError(f"SQL 执行失败: {exc}") from exc

    truncated = len(fetched) > limit
    rows_raw = fetched[:limit]
    rows: list[dict[str, Any]] = []
    for row in rows_raw:
        mapping = dict(row._mapping)  # noqa: SLF001
        # JSON 友好
        rows.append({k: _jsonable(v) for k, v in mapping.items()})

    return {
        "ok": True,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "sql": safe_sql,
    }


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    # datetime / Decimal / bytes
    try:
        from datetime import date, datetime
        from decimal import Decimal

        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    return str(value)


READONLY_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_readonly_sql",
        "description": (
            "通用只读 SQL（SELECT/WITH）。"
            "查门店扫码胎号明细请改用 query_scan_tyre_records（自动分表）；"
            "禁止查逻辑表 t_scan_store_record_tyre / t_scan_supplier_record_tyre / t_tyre（直连会 0 行）。"
            "查 K1 先 t_shop_info(shop_code)；查 K2 先 t_store_info(org_code)，"
            "用 account_num 作为 parent_code 再算 K2 扫码分表。"
            "本工具用于主数据、连锁关系、K1 扫码物理表、胎号主档物理表等；不要用于改数。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "单条 SELECT 或 WITH ... SELECT，不要分号结尾",
                },
                "reason": {
                    "type": "string",
                    "description": "简要说明为何需要查库（给系统日志用）",
                },
            },
            "required": ["sql"],
        },
    },
}
