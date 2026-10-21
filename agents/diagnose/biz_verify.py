"""业务核对：是否必须查库、扫码手册、连锁场景判定。"""

from __future__ import annotations

import re

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

# 固定场景：关键路径补检索词（与业务问句并行；避免带类名污染向量）
CHAIN_PATH_QUERIES: list[str] = [
    "isFalseClaim createOut KTWO_IN chain store",
    "入库是否冒领 isFalseClaim createOut",
    "if isChainStore KTWO_IN createOut return false",
    "k2InAndOrgCodeNotEquals createIn createOut chain store",
]

SQL_COOKBOOK = """\
【业务只读库核对手册——推理用，勿写入对用户回复】
- 门店扫码胎号明细：分表 `t_scan_store_record_tyre_0` … `t_scan_store_record_tyre_N`（只读库无逻辑表时需查分表或 UNION ALL）。
- 常用字段：`TYRE_CODE`、`ORG_CODE`、`PARENT_CODE`、`SCAN_TYPE`、`RECORD_TYPE`、`is_auto`、`CREATE_TIME`。
- 不确定列名时先查 `information_schema.COLUMNS`。
- 判定要点：
  1) `is_auto=1` 表示系统/连锁自动入出库，不是门店手工操作；
  2) A 零出 `is_auto=1` 且 B 入库 `is_auto=0` → 多为 B 主动入库触发给 A 自动出库；
  3) 若另有 B 的零出 → 更像 B 零售出库触发的连锁补账；
  4) 截图同一秒出现「A 零出 + B 入库」是连锁补账特征，须用 `is_auto` 区分谁触发。
- 查库失败须在结论中说明未能核对，禁止只凭截图断言自动/手工。"""


def _combined(*parts: str) -> str:
    return "\n".join(p for p in parts if (p or "").strip())


def needs_biz_sql_verify(user_text: str = "", image_description: str = "") -> bool:
    """胎号/入出库/连锁等业务核对题：必须查库。"""
    text = _combined(user_text, image_description)
    if not text.strip():
        return False
    if _TYRE_CODE_RE.search(text):
        return True
    return bool(_SCAN_BIZ_RE.search(text))


def looks_like_chain_scan_scene(user_text: str = "", image_description: str = "") -> bool:
    """连锁 / K2 自动入出库场景：触发关键路径补检索。"""
    text = _combined(user_text, image_description)
    if not text.strip():
        return False
    return bool(_CHAIN_SCENE_RE.search(text))


def sql_cookbook_suffix() -> str:
    return "\n\n" + SQL_COOKBOOK
