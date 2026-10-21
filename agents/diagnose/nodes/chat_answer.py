"""对话回答节点：基于 checkpoint 中的 messages 调用 LLM。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.diagnose.state import ChatAgentState
from app.core.config import get_settings
from app.services import llm_service

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是 TraceMind（溯源大脑）助手。"
    "结合对话历史，用简洁中文回答；不要编造未出现的事实。"
)


def _message_to_role_content(msg: BaseMessage) -> dict[str, str] | None:
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
    if role in ("human", "user"):
        return {"role": "user", "content": content}
    if role in ("ai", "assistant"):
        return {"role": "assistant", "content": content}
    if role == "system":
        return {"role": "system", "content": content}
    return None


def messages_for_llm(
    messages: list[Any],
    *,
    max_messages: int | None = None,
) -> list[dict[str, str]]:
    """LangChain messages → OpenAI chat messages（截断最近 N 条）。"""
    cfg = get_settings()
    limit = max_messages if max_messages is not None else cfg.agent_memory_max_messages
    converted: list[dict[str, str]] = []
    for msg in messages or []:
        item = _message_to_role_content(msg) if isinstance(msg, BaseMessage) else None
        if item and item["content"].strip():
            converted.append(item)
    if limit > 0 and len(converted) > limit:
        converted = converted[-limit:]
    return converted


def answer_node(state: ChatAgentState) -> dict[str, Any]:
    """同步回答：读全部 messages，追加 AIMessage。"""
    cfg = get_settings()
    history = messages_for_llm(list(state.get("messages") or []))
    openai_messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM},
        *history,
    ]
    errors: list[str] = list(state.get("errors") or [])
    reply = ""
    model_key = "qwen2.5-7b"
    pipeline = "agent:qwen2.5-7b+memory"

    try:
        result = llm_service.chat_messages(
            openai_messages,
            model_key="qwen2.5-7b",
            temperature=None,
            settings=cfg,
        )
        reply = result.get("reply") or ""
        model_key = result.get("key") or model_key
    except Exception as exc:  # noqa: BLE001
        logger.warning("qwen 失败，尝试 gpt-sol: %s", exc)
        errors.append(f"qwen2.5-7b: {exc}")
        try:
            result = llm_service.chat_messages(
                openai_messages,
                model_key="gpt-sol",
                temperature=None,
                settings=cfg,
            )
            reply = result.get("reply") or ""
            model_key = "gpt-sol"
            pipeline = "agent:fallback:gpt-sol+memory"
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"gpt-sol: {exc2}")
            reply = f"模型调用失败：{exc2}"
            pipeline = "agent:failed"

    return {
        "messages": [AIMessage(content=reply)],
        "reply": reply,
        "pipeline": pipeline,
        "model_key": model_key,
        "errors": errors,
    }


def build_human_content(
    user_text: str,
    *,
    image_description: str = "",
) -> str:
    """把 OCR 结果拼进本轮用户内容（写入 checkpoint，不再存原图）。"""
    text = (user_text or "").strip()
    ocr = (image_description or "").strip()
    if not ocr:
        return text or "（用户未提供文字）"
    return (
        "下面是图片 OCR 识别出的文字与用户原始问题，请综合回答。"
        "请以 OCR 文字为准，不要编造图中不存在的内容。\n\n"
        f"【图片OCR文字】\n{ocr}\n\n"
        f"【用户问题】\n{text or '（用户未提供文字，请根据OCR文字总结）'}"
    )


__all__ = [
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "answer_node",
    "messages_for_llm",
    "build_human_content",
    "_SYSTEM",
]
