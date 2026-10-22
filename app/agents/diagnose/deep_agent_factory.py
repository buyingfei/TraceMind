"""TraceMind Deep Agent：create_deep_agent + Agent Skills + 业务查库工具。"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.agents.tools.biz_lc_tools import list_biz_lc_tools
from app.agents.tools.readonly_sql import biz_readonly_enabled
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


def build_biz_tools() -> list[BaseTool]:
    """业务工具：来自 @tool 定义（app.agents.tools.biz_lc_tools）。"""
    if not biz_readonly_enabled():
        return []
    return list_biz_lc_tools()


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
    tools = build_biz_tools()

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
