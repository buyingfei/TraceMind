"""TraceMind Deep Agent：create_deep_agent + Agent Skills + 业务查库工具。"""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.agents.tools.readonly_sql import biz_readonly_enabled, run_select
from app.agents.tools.scan_tyre_query import query_scan_tyre_records
from app.agents.tools.warranty_query import query_warranty_by_plate
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_PROFILE_LOCK = threading.Lock()
_PROFILE_REGISTERED = False

# 相对 FilesystemBackend root 的 POSIX 路径（须以 / 开头）
SKILLS_SOURCE = "/app/agents/skills/"


def _ensure_harness_profile() -> None:
    """注册 openai 提供方 HarnessProfile：禁写盘/执行/子代理，仅保留读技能所需工具。"""
    global _PROFILE_REGISTERED
    if _PROFILE_REGISTERED:
        return
    with _PROFILE_LOCK:
        if _PROFILE_REGISTERED:
            return
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            register_harness_profile,
        )

        register_harness_profile(
            "openai",
            HarnessProfile(
                excluded_tools=frozenset(
                    {
                        "write_file",
                        "edit_file",
                        "delete",
                        "execute",
                        "task",
                    }
                ),
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            ),
        )
        _PROFILE_REGISTERED = True
        logger.info("deep agent harness profile registered for provider=openai")


def build_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    cfg = settings or get_settings()
    kwargs: dict[str, Any] = {
        "model": cfg.openai_model,
        "api_key": cfg.openai_api_key or "EMPTY",
        "base_url": cfg.normalized_openai_base_url(),
        "temperature": cfg.openai_temperature,
    }
    # 公司网关走 Chat Completions，避免默认 Responses API
    try:
        return ChatOpenAI(**kwargs, use_responses_api=False)
    except TypeError:
        return ChatOpenAI(**kwargs)


def _json_tool_result(data: dict[str, Any], *, max_chars: int = 6000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "…(truncated)"
    return text


def build_biz_tools() -> list[StructuredTool]:
    """与 tool_loop 同源的业务工具（LangChain StructuredTool）。"""

    def _scan(tyre_code: str, parent_code: str = "", reason: str = "") -> str:
        if reason:
            logger.info("deep-agent query_scan_tyre reason=%s", reason[:160])
        parent = (parent_code or "").strip() or None
        try:
            payload = query_scan_tyre_records(tyre_code, parent_code=parent)
            payload.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001
            payload = {"ok": False, "error": str(exc)}
        return _json_tool_result(payload)

    def _warranty(license_plate: str, reason: str = "") -> str:
        if reason:
            logger.info("deep-agent query_warranty reason=%s", reason[:160])
        try:
            payload = query_warranty_by_plate(license_plate)
            payload.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001
            payload = {"ok": False, "error": str(exc)}
        return _json_tool_result(payload)

    def _sql(sql: str, reason: str = "") -> str:
        if reason:
            logger.info("deep-agent readonly_sql reason=%s", reason[:200])
        try:
            payload = run_select(sql)
            if "error" not in payload:
                payload.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001
            payload = {"ok": False, "error": str(exc)}
        return _json_tool_result(payload)

    tools = [
        StructuredTool.from_function(
            func=_scan,
            name="query_scan_tyre_records",
            description=(
                "查询门店扫码胎号明细（自动按 parent_code 计算分表 t_scan_store_record_tyre_N）。"
                "单次超时约 5 分钟，失败自动重试 3 次。"
                "核对胎号流水、is_auto、门店入出库时优先用本工具；"
                "不要直接查逻辑表 t_scan_store_record_tyre。"
                "已知 K1/parent_code 请传入；未知时可只传 tyre_code。"
            ),
        ),
        StructuredTool.from_function(
            func=_warranty,
            name="query_warranty_by_plate",
            description=(
                "按车牌查询质保卡（t_warranty_detail）。"
                "单次超时约 5 分钟，失败自动重试 3 次。"
                "用户问某车牌有没有质保卡/对应胎号时必须用本工具；不要先查扫码流水。"
            ),
        ),
    ]
    if biz_readonly_enabled():
        tools.append(
            StructuredTool.from_function(
                func=_sql,
                name="run_readonly_sql",
                description=(
                    "通用只读 SQL（SELECT/WITH）。单次超时约 5 分钟，失败自动重试 3 次。"
                    "查门店扫码胎号明细请改用 query_scan_tyre_records；"
                    "禁止查逻辑表。查 K1 先 t_shop_info；查 K2 先 t_store_info。"
                    "执行策略见 skill biz-sql-query。"
                ),
            )
        )
    return tools


@lru_cache(maxsize=4)
def get_deep_agent(*, system_prompt: str, model_key: str = "gpt-sol"):
    """按 system_prompt 缓存 CompiledStateGraph（audience 不同则 prompt 不同）。"""
    from deepagents import FilesystemPermission, create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend

    _ensure_harness_profile()
    cfg = get_settings()
    # app/agents/diagnose/this_file.py → 仓库根 TraceMind
    root = Path(__file__).resolve().parents[3]
    skills_dir = root / "app" / "agents" / "skills"
    if not skills_dir.is_dir():
        raise RuntimeError(f"skills 目录不存在: {skills_dir}")

    backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    model = build_chat_model(cfg)
    tools = build_biz_tools() if biz_readonly_enabled() else []

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        skills=[SKILLS_SOURCE],
        backend=backend,
        permissions=[
            # 先允许读 skills，再拒绝其余读写（规则按声明顺序，首条命中生效）
            FilesystemPermission(
                operations=["read"],
                paths=["/app/agents/skills/**"],
                mode="allow",
            ),
            FilesystemPermission(operations=["read"], paths=["/**"], mode="deny"),
            FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
        ],
        name=f"tracemind-{model_key}",
    )
    logger.info(
        "deep agent created model=%s tools=%s skills=%s",
        cfg.openai_model,
        [t.name for t in tools],
        SKILLS_SOURCE,
    )
    return agent


def clear_deep_agent_cache() -> None:
    get_deep_agent.cache_clear()
