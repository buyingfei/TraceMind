"""对话回答节点：基于记忆 + 代码证据调用 LLM。"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.diagnose.biz_verify import needs_biz_sql_verify, sql_cookbook_suffix
from agents.diagnose.nodes.retrieve_code import format_code_evidence
from agents.diagnose.state import ChatAgentState
from agents.diagnose.tool_loop import chat_with_optional_tools
from agents.tools.readonly_sql import biz_readonly_enabled
from app.core.config import get_settings
from app.services import llm_service

logger = logging.getLogger(__name__)

# 业务回复中剥离常见代码定位痕迹
_CODE_PATH_RE = re.compile(
    r"(?m)^[ \t]*[-*•]?\s*`?(?:[\w./\\-]+\.(?:java|xml|yml|yaml|kt|ts|js|py))`?"
    r"(?:\s*[:：]\s*\d+(?:\s*[-–—]\s*\d+)?)?`?\s*$"
)
_INLINE_JAVA_LOC_RE = re.compile(
    r"`?[\w./\\-]+\.(?:java|xml|yml|yaml|kt)\s*:\s*\d+(?:\s*[-–—]\s*\d+)?`?"
)
_SYMBOL_LINE_RE = re.compile(
    r"(?m)^[ \t]*[-*•]?\s*(?:symbol|方法|类名|文件)\s*[=:：]\s*.+$",
    re.IGNORECASE,
)
_JAVA_IDENT_RE = re.compile(
    r"\b(?:isFalseClaim|isChainStore|createOut|createIn|k2InAndOrgCodeNotEquals|"
    r"queryScanStoreTyre|RetailCommService|TyreStatusVerifierService|"
    r"StoreRelationShip)\b"
)


def resolve_answer_audience(settings=None) -> str:
    cfg = settings or get_settings()
    audience = (cfg.answer_audience or "business").strip().lower()
    if audience not in ("business", "technical"):
        return "business"
    return audience


_SYSTEM_BUSINESS = (
    "你是 TraceMind（溯源大脑）助手，面向业务同学回答。"
    "推理阶段可以充分使用内部代码检索证据（含类名、方法名、路径）与查库结果，"
    "用来判断触发方与业务路径；但最终给用户的回复必须是纯业务语言。"
    "最终回复严禁出现：文件路径、.java/.xml 等扩展名、行号、类名、方法名、"
    "symbol、包名、Git 路径、原始 SQL、表名、字段名。"
    "涉及胎号流水、门店入出库、连锁归属、是否自动入出库时："
    "必须先调用 run_readonly_sql 查业务只读库核对（尤其 is_auto），再下结论；"
    "查库失败须明确说明未能用库核对。"
)

_FORMAT_BUSINESS = (
    "请按以下结构输出最终回复（不要出现「技术依据」「代码引用」等章节）：\n"
    "1. 业务结论：是否异常/是否符合系统规则；点明是谁触发、谁的操作是系统自动补账\n"
    "2. 原因说明：用业务语言解释（可提连锁门店、自动入出库、库存归属转移）；"
    "区分「B 主动入库触发 A 自动出」与「B 零售触发连锁补入出」\n"
    "3. 与截图/流水对齐：门店、操作类型、时间；可描述「系统自动」但不要写字段名\n"
    "须先查库再写上述结构；查库后仍不足则标注不确定。"
)

_SYSTEM_TECHNICAL = (
    "你是 TraceMind（溯源大脑）助手。"
    "结合对话历史、截图 OCR 事实与代码检索证据，用简洁中文回答；"
    "不要编造未出现在证据或对话中的事实。"
    "若有代码证据，必须在「技术依据」中引用 path:行号 或方法名。"
    "涉及胎号/门店入出库时，必须先 run_readonly_sql 查只读库（含 is_auto）再结论。"
)

_FORMAT_TECHNICAL = (
    "请按以下结构回答：\n"
    "1. 业务结论：用业务语言说明原因（是否故障/是否自动归属转移；谁触发）\n"
    "2. 技术依据：引用检索到的代码（path:行号 / symbol），说明关键判断与入出库调用\n"
    "3. 与截图对齐：若有 OCR/流水，点明对应门店与操作类型（如 K2零出→K2入库）\n"
    "数据类问题必须先查库；"
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


def build_system_prompt(*, audience: str | None = None) -> str:
    mode = audience or resolve_answer_audience()
    if mode == "technical":
        return "\n".join([_SYSTEM_TECHNICAL, _FORMAT_TECHNICAL])
    return "\n".join([_SYSTEM_BUSINESS, _FORMAT_BUSINESS])


def build_evidence_user_suffix(
    *,
    code_hits: list[dict[str, Any]] | None,
    audience: str | None = None,
    require_sql: bool = False,
) -> str:
    """检索证据注入：过程可用完整代码；business 仅约束最终回复不复述定位。"""
    hits = code_hits or []
    mode = audience or resolve_answer_audience()
    parts: list[str] = []

    if mode == "technical":
        evidence = format_code_evidence(hits)
        parts.append(f"\n\n【代码检索证据】\n{evidence}")
    elif not hits:
        parts.append(
            "\n\n【内部参考】\n"
            "（未检索到相关实现；请主要依据截图、查库结果与用户描述作答，并标注不确定。）"
        )
    else:
        evidence = format_code_evidence(hits)
        parts.append(
            "\n\n【内部实现参考——可在推理中使用类名/方法/路径，禁止写入最终对用户回复】\n"
            f"{evidence}\n"
            "最终回复请改写成业务语言，不要出现上述路径、行号、类名或方法名。"
        )

    if require_sql and biz_readonly_enabled():
        parts.append(sql_cookbook_suffix())
        parts.append(
            "\n\n【硬性要求】本题必须先调用 run_readonly_sql 查库核对后再写最终回答。"
        )

    return "".join(parts)


def sanitize_reply_for_audience(reply: str, *, audience: str | None = None) -> str:
    """business 模式下剥离路径/行号/关键方法名等代码痕迹。"""
    text = reply or ""
    mode = audience or resolve_answer_audience()
    if mode != "business" or not text.strip():
        return text

    cleaned = _CODE_PATH_RE.sub("", text)
    cleaned = _INLINE_JAVA_LOC_RE.sub("", cleaned)
    cleaned = _SYMBOL_LINE_RE.sub("", cleaned)
    cleaned = _JAVA_IDENT_RE.sub("系统逻辑", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]*#{0,3}\s*技术依据\s*[:：]?\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]*\d+\.\s*技术依据\s*[:：]?\s*$", "", cleaned)
    cleaned = re.sub(r"\b(?:t_scan_store_record_tyre(?:_\d+)?|is_auto|SCAN_TYPE)\b", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or text.strip()


def build_openai_messages_for_turn(
    *,
    history_messages: list[Any],
    human_content: str,
    code_hits: list[dict[str, Any]] | None = None,
    audience: str | None = None,
    require_sql: bool | None = None,
    user_text: str = "",
    image_description: str = "",
) -> list[dict[str, str]]:
    """组装含证据的多轮 messages（供流式/同步共用）。"""
    mode = audience or resolve_answer_audience()
    history = messages_for_llm(history_messages)
    must_sql = (
        require_sql
        if require_sql is not None
        else needs_biz_sql_verify(user_text or human_content, image_description)
    )
    user_content = human_content + build_evidence_user_suffix(
        code_hits=code_hits,
        audience=mode,
        require_sql=bool(must_sql and biz_readonly_enabled()),
    )
    return [
        {"role": "system", "content": build_system_prompt(audience=mode)},
        *history,
        {"role": "user", "content": user_content},
    ]


def answer_node(state: ChatAgentState) -> dict[str, Any]:
    """同步回答：读 messages + code_hits，追加 AIMessage。"""
    cfg = get_settings()
    audience = resolve_answer_audience(cfg)
    history_all = list(state.get("messages") or [])
    history_without_last_human = history_all
    last_human = ""
    if history_all:
        last = history_all[-1]
        if getattr(last, "type", None) in ("human", "user"):
            last_human = str(getattr(last, "content", "") or "")
            history_without_last_human = history_all[:-1]

    human = last_human or str(state.get("user_text") or "")
    user_text = str(state.get("user_text") or human)
    image_description = str(state.get("image_description") or "")
    code_hits = list(state.get("code_hits") or [])
    require_sql = needs_biz_sql_verify(user_text, image_description) and biz_readonly_enabled()
    openai_messages = build_openai_messages_for_turn(
        history_messages=history_without_last_human,
        human_content=human,
        code_hits=code_hits,
        audience=audience,
        require_sql=require_sql,
        user_text=user_text,
        image_description=image_description,
    )

    errors: list[str] = list(state.get("errors") or [])
    reply = ""
    model_key = "gpt-sol"
    used_sql = False
    pipeline = "agent:retrieve->gpt-sol+memory"

    try:
        result = chat_with_optional_tools(
            openai_messages,
            model_key="gpt-sol",
            settings=cfg,
            require_sql=require_sql,
        )
        reply = sanitize_reply_for_audience(
            result.get("reply") or "",
            audience=audience,
        )
        model_key = result.get("key") or model_key
        used_sql = bool(result.get("used_tools"))
        if used_sql:
            pipeline = "agent:retrieve->sql->gpt-sol+memory"
        elif require_sql:
            pipeline = "agent:retrieve->gpt-sol+sql-required+memory"
            errors.append("require_sql: 模型未成功完成查库核对")
        elif biz_readonly_enabled():
            pipeline = "agent:retrieve->gpt-sol+tools+memory"
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpt-sol 重试后仍失败: %s", exc)
        errors.append(f"gpt-sol: {exc}")
        reply = f"模型调用失败：{exc}"
        pipeline = "agent:failed"

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
    "sanitize_reply_for_audience",
    "resolve_answer_audience",
]
