"""按分片查询门店扫码胎号明细（t_scan_store_record_tyre_*）。"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import text

from app.agents.tools.biz_sharding import (
    SCAN_STORE_TYRE_SHARD_COUNT,
    all_scan_store_tyre_tables,
    scan_store_tyre_table,
    store_scan_shard_index,
)
from app.agents.tools.readonly_sql import _jsonable
from app.agents.tools.sql_exec import apply_session_timeout, run_with_sql_retry
from app.infra.biz_db import biz_readonly_enabled, get_biz_readonly_engine
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_TYRE_RE = re.compile(r"\b(\d{8,14})\b")
_PARENT_RE = re.compile(r"\b(CKH\w+|IZX\w+)\b", re.IGNORECASE)

# 业务侧常见含义（供模型推理；勿原样甩给用户）
_RECORD_TYPE_HINT = {
    1: "出库/零出类",
    2: "入库类",
}


def extract_tyre_code(*texts: str) -> str | None:
    for t in texts:
        m = _TYRE_RE.search(t or "")
        if m:
            return m.group(1)
    return None


def extract_parent_code(*texts: str) -> str | None:
    for t in texts:
        m = _PARENT_RE.search(t or "")
        if m:
            return m.group(1).upper()
    return None


def _select_cols() -> str:
    return (
        "TYRE_CODE, ORG_CODE, PARENT_CODE, SCAN_TYPE, RECORD_TYPE, is_auto, "
        "CREATE_TIME, RECORD_CODE, STATUS, SKU_CODE"
    )


def _query_one_shard(
    conn,
    table: str,
    tyre_code: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    sql = (
        f"SELECT {_select_cols()} FROM `{table}` "
        "WHERE TYRE_CODE = :tyre ORDER BY CREATE_TIME ASC LIMIT :lim"
    )
    rows = conn.execute(text(sql), {"tyre": tyre_code, "lim": limit}).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {k: _jsonable(v) for k, v in dict(row).items()}
        rt = item.get("RECORD_TYPE")
        try:
            item["record_type_hint"] = _RECORD_TYPE_HINT.get(int(rt), "未知")
        except (TypeError, ValueError):
            item["record_type_hint"] = "未知"
        item["shard_table"] = table
        out.append(item)
    return out


def query_scan_tyre_records(
    tyre_code: str,
    *,
    parent_code: str | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """按 parent_code 算分表查询；无 parent 时扫全部分表兜底。"""
    if not biz_readonly_enabled():
        raise RuntimeError("业务只读 SQL 未启用或未配置连接串")

    tyre = (tyre_code or "").strip()
    if not tyre:
        raise ValueError("tyre_code 不能为空")
    parent = (parent_code or "").strip() or None

    cfg = get_settings()
    limit = max_rows if max_rows is not None else int(cfg.biz_readonly_max_rows)
    limit = max(1, min(limit, 200))

    def _once() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        engine = get_biz_readonly_engine()
        route: dict[str, Any] = {
            "tyre_code": tyre,
            "parent_code": parent,
            "shard_count": SCAN_STORE_TYRE_SHARD_COUNT,
            "mode": None,
            "shard_index": None,
            "shard_table": None,
        }
        with engine.connect() as conn:
            apply_session_timeout(conn)
            rows: list[dict[str, Any]] = []
            if parent:
                table = scan_store_tyre_table(parent)
                idx = store_scan_shard_index(parent)
                route.update(
                    mode="by_parent_code",
                    shard_index=idx,
                    shard_table=table,
                )
                try:
                    rows = _query_one_shard(conn, table, tyre, limit=limit)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("分表查询失败 table=%s: %s", table, exc)
                    raise RuntimeError(f"分表查询失败 {table}: {exc}") from exc
            else:
                route["mode"] = "scan_all_shards"
                for table in all_scan_store_tyre_tables():
                    try:
                        part = _query_one_shard(conn, table, tyre, limit=limit)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("skip shard %s: %s", table, exc)
                        continue
                    if part:
                        rows.extend(part)
                        if len(rows) >= limit:
                            break
                rows = sorted(rows, key=lambda r: str(r.get("CREATE_TIME") or ""))[:limit]
                if rows:
                    route["shard_table"] = rows[0].get("shard_table")
                    route["parent_code"] = rows[0].get("PARENT_CODE") or parent
                    if route["parent_code"]:
                        try:
                            route["shard_index"] = store_scan_shard_index(
                                str(route["parent_code"])
                            )
                        except Exception:  # noqa: BLE001
                            pass
        return rows, route

    rows, route = run_with_sql_retry(_once, label="query_scan_tyre")

    return {
        "ok": True,
        "row_count": len(rows),
        "rows": rows,
        "route": route,
        "hint": (
            "is_auto=1 表示系统/连锁自动入出库；"
            "RECORD_TYPE 1≈出库/零出，2≈入库。"
            "若 A 零出 is_auto=1 且 B 入库 is_auto=0，多为 B 主动入库触发 A 自动出库。"
        ),
    }


# OpenAI schema / @tool 定义见 app.agents.tools.biz_lc_tools
