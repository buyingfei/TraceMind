"""按车牌查询质保卡（t_warranty_detail）。"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import text

from app.agents.tools.readonly_sql import _jsonable
from app.infra.biz_db import biz_readonly_enabled, get_biz_readonly_engine
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 简易车牌：省份简称 + 字母数字
_PLATE_RE = re.compile(
    r"([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
    r"[A-HJ-NP-Z][A-HJ-NP-Z0-9]{4,6})"
)


def extract_license_plate(*texts: str) -> str | None:
    for t in texts:
        m = _PLATE_RE.search((t or "").upper().replace(" ", ""))
        if m:
            return m.group(1)
        # 兼容原文大小写混排
        m2 = _PLATE_RE.search(t or "")
        if m2:
            return m2.group(1)
    return None


def normalize_plate(plate: str) -> str:
    return re.sub(r"\s+", "", (plate or "").strip()).upper()


def query_warranty_by_plate(license_plate: str, *, max_rows: int | None = None) -> dict[str, Any]:
    """按车牌精确查质保明细。"""
    if not biz_readonly_enabled():
        raise RuntimeError("业务只读 SQL 未启用或未配置连接串")

    plate = normalize_plate(license_plate)
    if not plate:
        raise ValueError("license_plate 不能为空")

    cfg = get_settings()
    limit = max_rows if max_rows is not None else int(cfg.biz_readonly_max_rows)
    limit = max(1, min(limit, 200))
    # 该表 license_plate 无索引，精确查询也可能较慢，单独放宽超时
    timeout = max(60, int(cfg.biz_readonly_timeout_sec))

    # 不 ORDER BY，避免大表无索引时额外排序拖垮超时
    sql = (
        "SELECT WARRANTY_CARD, TYRE_CODE, license_plate, WARRANTY_STATUS, "
        "STORE_CODE, ORG_CODE, ORG_PARENT, BRAND, PATTERN, OWNER_NAME, "
        "OWNER_MOBILE, ACQUIRE_TIME, EFFECTIVE_BEGIN, EFFECTIVE_END, CREATE_TIME "
        "FROM t_warranty_detail "
        "WHERE license_plate = :plate "
        "LIMIT :lim"
    )

    engine = get_biz_readonly_engine()
    with engine.connect() as conn:
        try:
            conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={timeout * 1000}"))
        except Exception:  # noqa: BLE001
            pass
        try:
            rows_raw = conn.execute(text(sql), {"plate": plate, "lim": limit}).mappings().all()
        except Exception as exc:  # noqa: BLE001
            logger.warning("质保卡按车牌查询失败 plate=%s: %s", plate, exc)
            raise RuntimeError(
                f"质保卡查询超时或失败（t_warranty_detail.license_plate 可能无可用索引）: {exc}"
            ) from exc

    rows = [{k: _jsonable(v) for k, v in dict(r).items()} for r in rows_raw]
    return {
        "ok": True,
        "row_count": len(rows),
        "rows": rows,
        "license_plate": plate,
        "table": "t_warranty_detail",
        "hint": (
            "有行即表示该车牌有质保卡记录；TYRE_CODE 为对应胎号，"
            "WARRANTY_CARD 为质保卡号，WARRANTY_STATUS 为状态。"
        ),
    }


QUERY_WARRANTY_BY_PLATE_TOOL = {
    "type": "function",
    "function": {
        "name": "query_warranty_by_plate",
        "description": (
            "按车牌号查询质保卡明细表 t_warranty_detail。"
            "用户问「某车牌有没有质保卡 / 对应胎号是什么」时必须用本工具；"
            "不要先查扫码流水，也不要使用上一轮对话里的胎号。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "license_plate": {
                    "type": "string",
                    "description": "车牌号，如 晋D019X0",
                },
                "reason": {
                    "type": "string",
                    "description": "查库目的（日志用）",
                },
            },
            "required": ["license_plate"],
        },
    },
}
