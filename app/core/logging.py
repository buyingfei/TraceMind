"""日志初始化与精简摘要工具。"""

from __future__ import annotations

import logging
import re


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


_WS_RE = re.compile(r"\s+")


def clip_text(text: str | None, *, limit: int = 120) -> str:
    """单行摘要，避免把大段模型/OCR 全文打进日志。"""
    raw = _WS_RE.sub(" ", (text or "").strip())
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 1)] + "…"


def summarize_messages(messages: list[dict] | None, *, preview: int = 80) -> str:
    """统计 messages 条数与角色，附最后一条 user 预览。"""
    msgs = messages or []
    roles: dict[str, int] = {}
    last_user = ""
    for m in msgs:
        role = str(m.get("role") or "?")
        roles[role] = roles.get(role, 0) + 1
        content = m.get("content")
        if role == "user" and isinstance(content, str):
            last_user = content
    role_part = ",".join(f"{k}:{v}" for k, v in sorted(roles.items()))
    return (
        f"n={len(msgs)} roles={{{role_part}}} "
        f"last_user={clip_text(last_user, limit=preview)!r}"
    )
