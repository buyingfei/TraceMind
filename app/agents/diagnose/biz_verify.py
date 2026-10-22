"""业务核对：意图识别、是否必须查库、手册文案。"""

from __future__ import annotations

import re
from typing import Literal

from app.agents.skills import skill_suffix_for_intent

BizIntent = Literal["scan_tyre", "warranty_plate", "generic_biz", "none"]

# 胎号常见为较长数字
_TYRE_CODE_RE = re.compile(r"\b\d{8,14}\b")
_SCAN_BIZ_RE = re.compile(
    r"(胎号|轮胎|入库|出库|零出|连锁|门店|库存|归属|扫码|K\s*1|K\s*2|"
    r"自动入|自动出|is_?auto|流水)",
    re.IGNORECASE,
)
_CHAIN_SCENE_RE = re.compile(
    r"(连锁|自动.*(?:入|出)库|K\s*2.*(?:入|出)|零出|归属转移|冒领|"
    r"isFalseClaim|createOut|isChainStore)",
    re.IGNORECASE,
)
_WARRANTY_RE = re.compile(r"(质保|保修卡|保修|warranty)", re.IGNORECASE)
_PLATE_HINT_RE = re.compile(
    r"(车牌|牌照|[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
    r"[A-HJ-NP-Z])",
    re.IGNORECASE,
)

# 固定场景：关键路径补检索词（与业务问句并行；避免带类名污染向量）
CHAIN_PATH_QUERIES: list[str] = [
    "isFalseClaim createOut KTWO_IN chain store",
    "入库是否冒领 isFalseClaim createOut",
    "if isChainStore KTWO_IN createOut return false",
    "k2InAndOrgCodeNotEquals createIn createOut chain store",
]

SQL_COOKBOOK_SCAN = """\
【扫码核对——推理用，勿写入对用户回复】
- 查胎号扫码明细：调用 `query_scan_tyre_records(tyre_code, parent_code?)`。
  分片键 `parent_code`，物理表 `t_scan_store_record_tyre_0..63`；禁止查逻辑表。
- K1 主数据先查 `t_shop_info`；K2 先查 `t_store_info`（`account_num`→所属 K1）。
- K1 扫码明细表族：`t_scan_supplier_record_tyre_*`，分片键 `org_code`(K1)。
- `is_auto=1` 为系统自动；A 零出自动且 B 入库非自动 → 多为 B 入库触发。
- 只用本轮用户消息中的胎号，禁止沿用上一轮对话的胎号/K1。
- 完整分表路由见 skill `biz-sharding-sql`（Deep Agent 会按需 read_file；经典模式注入正文）。"""

SQL_COOKBOOK_WARRANTY = """\
【质保卡核对——推理用，勿写入对用户回复】
- 按车牌查质保：必须调用 `query_warranty_by_plate(license_plate)`，表 `t_warranty_detail`。
- 返回字段含 WARRANTY_CARD、TYRE_CODE、license_plate、WARRANTY_STATUS。
- 不要先查扫码流水；不要使用上一轮对话里的胎号。"""

SQL_COOKBOOK = SQL_COOKBOOK_SCAN  # 兼容旧引用


def _combined(*parts: str) -> str:
    return "\n".join(p for p in parts if (p or "").strip())


def detect_biz_intent(user_text: str = "", image_description: str = "") -> BizIntent:
    """仅依据本轮用户问句/OCR 判断意图（不要混入历史）。"""
    text = _combined(user_text, image_description)
    if not text.strip():
        return "none"
    # 车牌+质保优先于泛「胎号」词（「如有，胎号是什么」仍属质保场景）
    if _WARRANTY_RE.search(text) or (
        _PLATE_HINT_RE.search(text) and ("卡" in text or "质保" in text or "胎号" in text)
    ):
        if _PLATE_HINT_RE.search(text) or _WARRANTY_RE.search(text):
            return "warranty_plate"
    if _CHAIN_SCENE_RE.search(text) or (
        _SCAN_BIZ_RE.search(text) and _TYRE_CODE_RE.search(text)
    ):
        return "scan_tyre"
    if _TYRE_CODE_RE.search(text) or _SCAN_BIZ_RE.search(text):
        return "scan_tyre"
    if _WARRANTY_RE.search(text) or _PLATE_HINT_RE.search(text):
        return "warranty_plate"
    return "none"


def needs_biz_sql_verify(user_text: str = "", image_description: str = "") -> bool:
    """本轮问题需要查业务库。"""
    return detect_biz_intent(user_text, image_description) != "none"


def looks_like_chain_scan_scene(user_text: str = "", image_description: str = "") -> bool:
    """连锁 / K2 自动入出库场景：触发关键路径补检索。"""
    text = _combined(user_text, image_description)
    if not text.strip():
        return False
    if detect_biz_intent(user_text, image_description) == "warranty_plate":
        return False
    return bool(_CHAIN_SCENE_RE.search(text))


def sql_cookbook_suffix(*, intent: BizIntent | None = None) -> str:
    """查库手册 + 按意图动态挂载 knowledge（分表规则等）。"""
    if intent == "warranty_plate":
        base = "\n\n" + SQL_COOKBOOK_WARRANTY
    elif intent == "scan_tyre":
        base = "\n\n" + SQL_COOKBOOK_SCAN
    else:
        base = "\n\n" + SQL_COOKBOOK_SCAN + "\n\n" + SQL_COOKBOOK_WARRANTY
    return base + skill_suffix_for_intent(intent)
