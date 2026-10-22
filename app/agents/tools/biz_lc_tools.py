"""业务查库工具：LangChain @tool 定义（Deep Agent / tool_loop / 换模型共用）。

实现仍在 scan_tyre_query / warranty_query / readonly_sql；本模块只做直观的工具壳。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agents.tools.readonly_sql import biz_readonly_enabled
from app.agents.tools.readonly_sql import run_select as _run_select
from app.agents.tools.scan_tyre_query import (
    query_scan_tyre_records as _run_query_scan_tyre,
)
from app.agents.tools.warranty_query import (
    query_warranty_by_plate as _run_query_warranty,
)

logger = logging.getLogger(__name__)


def _json_result(data: dict[str, Any], *, max_chars: int = 6000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "…(truncated)"
    return text


@tool
def query_scan_tyre_records(
    tyre_code: str,
    parent_code: str = "",
    reason: str = "",
) -> str:
    """查询门店扫码胎号明细（自动按 parent_code 计算分表 t_scan_store_record_tyre_N）。

    单次超时约 5 分钟，失败自动重试 3 次。核对胎号流水、is_auto、门店入出库时优先用本工具；
    不要直接查逻辑表 t_scan_store_record_tyre。已知 K1/parent_code 请传入；未知时可只传 tyre_code。

    Args:
        tyre_code: 胎号，如 9001462521
        parent_code: 分片键 K1/parent_code（如 CKH0024005）；可选
        reason: 简要说明查库目的（日志用）
    """
    if reason:
        logger.info("tool query_scan_tyre reason=%s tyre=%s", reason[:160], tyre_code)
    parent = (parent_code or "").strip() or None
    try:
        payload = _run_query_scan_tyre(tyre_code, parent_code=parent)
        payload.setdefault("ok", True)
    except Exception as exc:  # noqa: BLE001
        payload = {"ok": False, "error": str(exc)}
    return _json_result(payload)


@tool
def query_warranty_by_plate(license_plate: str, reason: str = "") -> str:
    """按车牌号查询质保卡明细表 t_warranty_detail。

    单次超时约 5 分钟，失败自动重试 3 次。用户问「某车牌有没有质保卡 / 对应胎号」时必须用本工具；
    不要先查扫码流水，也不要使用上一轮对话里的胎号。

    Args:
        license_plate: 车牌号，如 晋D019X0
        reason: 查库目的（日志用）
    """
    if reason:
        logger.info("tool query_warranty reason=%s plate=%s", reason[:160], license_plate)
    try:
        payload = _run_query_warranty(license_plate)
        payload.setdefault("ok", True)
    except Exception as exc:  # noqa: BLE001
        payload = {"ok": False, "error": str(exc)}
    return _json_result(payload)


@tool
def run_readonly_sql(sql: str, reason: str = "") -> str:
    """通用只读 SQL（SELECT/WITH）。单次超时约 5 分钟，失败自动重试 3 次。

    查门店扫码胎号明细请改用 query_scan_tyre_records；禁止查逻辑表。
    查 K1 先 t_shop_info；查 K2 先 t_store_info。执行策略见 skill biz-sql-query。

    Args:
        sql: 单条 SELECT 或 WITH ... SELECT，不要分号结尾
        reason: 简要说明为何需要查库（日志用）
    """
    if reason:
        logger.info("tool readonly_sql reason=%s", reason[:200])
    try:
        payload = _run_select(sql)
        if "error" not in payload:
            payload.setdefault("ok", True)
    except Exception as exc:  # noqa: BLE001
        payload = {"ok": False, "error": str(exc)}
    return _json_result(payload)


def list_biz_lc_tools(*, include_readonly_sql: bool | None = None) -> list[BaseTool]:
    """供 Deep Agent / bind_tools 使用的工具列表。"""
    enabled = biz_readonly_enabled() if include_readonly_sql is None else include_readonly_sql
    tools: list[BaseTool] = [query_scan_tyre_records, query_warranty_by_plate]
    if enabled:
        tools.append(run_readonly_sql)
    return tools


def openai_biz_tools(*, include_readonly_sql: bool | None = None) -> list[dict[str, Any]]:
    """经典 tool_loop：把 @tool 转成 OpenAI function calling schema。"""
    return [
        convert_to_openai_tool(t)
        for t in list_biz_lc_tools(include_readonly_sql=include_readonly_sql)
    ]


def biz_tool_by_name() -> dict[str, BaseTool]:
    return {t.name: t for t in list_biz_lc_tools(include_readonly_sql=True)}


# 兼容旧名（OpenAI schema 由 @tool 派生）
QUERY_SCAN_TYRE_TOOL = convert_to_openai_tool(query_scan_tyre_records)
QUERY_WARRANTY_BY_PLATE_TOOL = convert_to_openai_tool(query_warranty_by_plate)
READONLY_SQL_TOOL = convert_to_openai_tool(run_readonly_sql)
