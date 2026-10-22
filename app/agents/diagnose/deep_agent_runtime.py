"""Deep Agent 一轮推理：对齐 chat_with_optional_tools 返回结构。"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.diagnose.biz_verify import BizIntent, detect_biz_intent
from app.agents.diagnose.deep_agent_factory import get_deep_agent
from app.agents.diagnose.nodes.chat_answer import build_system_prompt, resolve_answer_audience
from app.agents.tools.scan_tyre_query import extract_parent_code, extract_tyre_code
from app.agents.tools.warranty_query import extract_license_plate
from app.core.config import get_settings
from app.core.logging import clip_text
from app.services import llm_service

logger = logging.getLogger(__name__)

MetaCallback = Any  # Callable[[dict[str, Any]], None]

_BIZ_TOOL_NAMES = frozenset(
    {
        "query_scan_tyre_records",
        "query_warranty_by_plate",
        "run_readonly_sql",
    }
)


def _openai_to_lc_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[Any]]:
    """拆出 system，其余转为 LangChain messages。"""
    system: str | None = None
    out: list[Any] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else str(content or "")
        if role == "system":
            system = text
            continue
        if role == "user":
            out.append(HumanMessage(content=text))
        elif role == "assistant":
            out.append(AIMessage(content=text))
        elif role == "tool":
            out.append(
                ToolMessage(
                    content=text,
                    tool_call_id=str(m.get("tool_call_id") or "unknown"),
                )
            )
    return system, out


def _tool_name_from_ai(msg: AIMessage) -> list[tuple[str, str]]:
    """返回 [(tool_name, arguments_json), ...]。"""
    pairs: list[tuple[str, str]] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        if isinstance(tc, dict):
            name = str(tc.get("name") or "")
            args = tc.get("args") or {}
        else:
            name = str(getattr(tc, "name", "") or "")
            args = getattr(tc, "args", {}) or {}
        if not name:
            continue
        if isinstance(args, str):
            raw = args
        else:
            raw = json.dumps(args, ensure_ascii=False, default=str)
        pairs.append((name, raw))
    return pairs


def _collect_traces(result_messages: list[Any]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}

    for msg in result_messages:
        if isinstance(msg, AIMessage):
            for name, raw in _tool_name_from_ai(msg):
                if name not in _BIZ_TOOL_NAMES and name not in {
                    "read_file",
                    "ls",
                    "glob",
                    "grep",
                }:
                    continue
                entry = {
                    "tool": name,
                    "arguments": raw[:500],
                    "ok": None,
                    "row_count": None,
                    "error": None,
                    "route": None,
                }
                traces.append(entry)
                # 粗略按顺序匹配后续 ToolMessage
                pending.setdefault(name, entry)
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
            # 回填最近一个尚无 ok 的同名/任意 pending
            target = None
            for t in reversed(traces):
                if t.get("ok") is None:
                    target = t
                    break
            if target is None:
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                target["ok"] = True
                continue
            if isinstance(payload, dict):
                target["ok"] = bool(payload.get("ok", True)) and not payload.get("error")
                target["row_count"] = payload.get("row_count")
                target["error"] = payload.get("error")
                target["route"] = payload.get("route")
            else:
                target["ok"] = True
    return traces


def _final_ai_text(result_messages: list[Any]) -> str:
    for msg in reversed(result_messages or []):
        if isinstance(msg, AIMessage):
            # 跳过仅含 tool_calls 的中间轮
            if getattr(msg, "tool_calls", None):
                content = msg.content
                text = content if isinstance(content, str) else ""
                if not (text or "").strip():
                    continue
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif isinstance(block, str):
                        parts.append(block)
                joined = "".join(parts).strip()
                if joined:
                    return joined
    return ""


def run_with_deep_agent(
    messages: list[dict[str, Any]],
    *,
    model_key: str = "gpt-sol",
    settings=None,
    on_meta: MetaCallback | None = None,
    require_sql: bool = False,
    current_user_text: str = "",
    current_image_description: str = "",
) -> dict[str, Any]:
    """调用 create_deep_agent 图；返回结构与 chat_with_optional_tools 对齐。"""
    cfg = settings or get_settings()
    audience = resolve_answer_audience(cfg)
    system_from_msgs, lc_messages = _openai_to_lc_messages(messages)
    system_prompt = system_from_msgs or build_system_prompt(audience=audience)

    turn_text = (current_user_text or "").strip()
    intent: BizIntent = detect_biz_intent(turn_text, current_image_description)
    inferred_tyre = extract_tyre_code(turn_text, current_image_description)
    inferred_parent = extract_parent_code(turn_text, current_image_description)
    inferred_plate = extract_license_plate(turn_text, current_image_description)

    if require_sql:
        if intent == "warranty_plate" and inferred_plate:
            lc_messages.append(
                HumanMessage(
                    content=(
                        f"请先调用 query_warranty_by_plate，"
                        f"参数 license_plate={inferred_plate}。"
                        "不要使用上一轮对话的胎号，不要查扫码流水。"
                        "若问题涉及胎号/K1/K2/扫码分表，先按 skills 中 biz-sharding-sql 路由。"
                    )
                )
            )
        elif intent == "scan_tyre" and inferred_tyre:
            hint = (
                f"请先调用 query_scan_tyre_records，参数 tyre_code={inferred_tyre}"
            )
            if inferred_parent:
                hint += f", parent_code={inferred_parent}"
            hint += (
                "。只用本轮胎号。"
                "分表与主数据路由请遵循 skill `biz-sharding-sql`（必要时 read_file 读取）。"
            )
            lc_messages.append(HumanMessage(content=hint))
        else:
            lc_messages.append(
                HumanMessage(
                    content=(
                        "本题必须先查业务只读库核对。"
                        "涉及胎号/K1/K2/扫码时请激活 skill `biz-sharding-sql` 后再写 SQL 或调工具。"
                    )
                )
            )

    if on_meta:
        on_meta(
            {
                "type": "meta",
                "stage": "sql_query",
                "message": "Deep Agent 正在推理（skills + 查库工具）…",
                "runtime": "deep_agent",
            }
        )

    agent = get_deep_agent(system_prompt=system_prompt, model_key=model_key)
    logger.info(
        "deep-agent invoke intent=%s require_sql=%s msgs=%s tyre=%s plate=%s",
        intent,
        require_sql,
        len(lc_messages),
        inferred_tyre,
        inferred_plate,
    )

    result = agent.invoke({"messages": lc_messages})
    result_messages = list(result.get("messages") or [])
    traces = _collect_traces(result_messages)
    reply = _final_ai_text(result_messages)
    used_tools = any(t.get("tool") in _BIZ_TOOL_NAMES for t in traces)

    if on_meta and traces:
        biz_traces = [t for t in traces if t.get("tool") in _BIZ_TOOL_NAMES]
        last = biz_traces[-1] if biz_traces else traces[-1]
        on_meta(
            {
                "type": "meta",
                "stage": "sql_done",
                "message": (
                    f"查库完成，返回 {last.get('row_count', 0)} 行"
                    if last.get("ok")
                    else f"工具结束：{last.get('tool')}"
                ),
                "row_count": last.get("row_count"),
                "tool": last.get("tool"),
                "runtime": "deep_agent",
            }
        )

    model_name = llm_service._resolve_model_name(model_key, cfg)  # noqa: SLF001
    logger.info(
        "deep-agent done used_tools=%s traces=%s preview=%r",
        used_tools,
        len(traces),
        clip_text(reply, limit=100),
    )
    return {
        "reply": reply,
        "model": model_name,
        "key": model_key,
        "tool_traces": traces,
        "used_tools": used_tools,
        "require_sql": require_sql,
        "intent": intent,
        "runtime": "deep_agent",
    }
