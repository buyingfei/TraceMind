"""对话回答节点：基于记忆 + 代码证据调用 LLM。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.diagnose.nodes.retrieve_code import format_code_evidence
from agents.diagnose.state import ChatAgentState
from app.core.config import get_settings
from app.services import llm_service

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是 TraceMind（溯源大脑）助手。"
    "结合对话历史、截图 OCR 事实与代码检索证据，用简洁中文回答；"
    "不要编造未出现在证据或对话中的事实。"
    "若有代码证据，必须在「技术依据」中引用 path:行号 或方法名。"
)

_ANSWER_FORMAT = (
    "请按以下结构回答：\n"
    "1. 业务结论：用业务语言说明原因（是否故障/是否自动归属转移）\n"
    "2. 技术依据：引用检索到的代码（path:行号 / symbol），说明关键判断与入出库调用\n"
    "3. 与截图对齐：若有 OCR/流水，点明对应门店与操作类型（如 K2零出→K2入库）\n"
    "若未命中代码，明确说明「未命中代码，请先完成索引」，仍可基于截图做有限推测并标注不确定。"
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


def build_system_prompt() -> str:
    return "\n".join([_SYSTEM, _ANSWER_FORMAT])


def build_evidence_user_suffix(
    *,
    code_hits: list[dict[str, Any]] | None,
) -> str:
    evidence = format_code_evidence(code_hits or [])
    return f"\n\n【代码检索证据】\n{evidence}"


def build_openai_messages_for_turn(
    *,
    history_messages: list[Any],
    human_content: str,
    code_hits: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """组装含证据的多轮 messages（供流式/同步共用）。"""
    history = messages_for_llm(history_messages)
    user_content = human_content + build_evidence_user_suffix(code_hits=code_hits)
    return [
        {"role": "system", "content": build_system_prompt()},
        *history,
        {"role": "user", "content": user_content},
    ]


def answer_node(state: ChatAgentState) -> dict[str, Any]:
    """同步回答：读 messages + code_hits，追加 AIMessage。"""
    cfg = get_settings()
    history_all = list(state.get("messages") or [])
    # 最后一条 human 已在 messages 中；证据拼进「覆盖」最后一轮提示：
    # 为避免重复，把历史去掉最后一条 human，再用带证据的 human 提问。
    history_without_last_human = history_all
    last_human = ""
    if history_all:
        last = history_all[-1]
        if getattr(last, "type", None) in ("human", "user"):
            last_human = str(getattr(last, "content", "") or "")
            history_without_last_human = history_all[:-1]

    human = last_human or str(state.get("user_text") or "")
    code_hits = list(state.get("code_hits") or [])
    openai_messages = build_openai_messages_for_turn(
        history_messages=history_without_last_human,
        human_content=human,
        code_hits=code_hits,
    )

    errors: list[str] = list(state.get("errors") or [])
    reply = ""
    model_key = "qwen2.5-7b"
    pipeline = "agent:retrieve->qwen2.5-7b+memory"

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
            pipeline = "agent:retrieve->fallback:gpt-sol+memory"
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"gpt-sol: {exc2}")
            reply = f"模型调用失败：{exc2}"
            pipeline = "agent:failed"

    if not code_hits and "未命中" not in reply:
        # 轻量提示（模型可能已说明）
        pass

    return {
        "messages": [AIMessage(content=reply)],
        "reply": reply,
        "pipeline": pipeline,
        "model_key": model_key,
        "errors": errors,
        "code_hits": code_hits,
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
    "build_openai_messages_for_turn",
    "build_system_prompt",
    "_SYSTEM",
]
