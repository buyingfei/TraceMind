"""公司模型 + 工具循环（扫码分表 / 质保车牌 / 通用 SQL）。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from agents.diagnose.biz_verify import BizIntent, detect_biz_intent
from agents.tools.readonly_sql import (
    READONLY_SQL_TOOL,
    biz_readonly_enabled,
    run_select,
)
from agents.tools.scan_tyre_query import (
    QUERY_SCAN_TYRE_TOOL,
    extract_parent_code,
    extract_tyre_code,
    query_scan_tyre_records,
)
from agents.tools.warranty_query import (
    QUERY_WARRANTY_BY_PLATE_TOOL,
    extract_license_plate,
    query_warranty_by_plate,
)
from app.core.config import get_settings
from app.services import llm_service

logger = logging.getLogger(__name__)

MetaCallback = Callable[[dict[str, Any]], None]

_FORCE_SQL_NUDGE = (
    "系统要求：本题必须先查业务只读库核对，且只用本轮用户给出的业务键"
    "（车牌/胎号），禁止沿用上一轮对话实体。请立即发起对应工具调用。"
)


def _tool_result_payload(data: dict[str, Any], *, max_chars: int = 6000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "…(truncated)"
    return text


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """只取本轮用户内容，避免历史胎号污染。"""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                # 证据后缀很长时，优先截取「用户问题」段
                if "【用户问题】" in content:
                    return content.split("【用户问题】", 1)[-1].strip()
                return content
    return ""


def run_readonly_sql_tool(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"sql": arguments}
    sql = str((arguments or {}).get("sql") or "").strip()
    reason = str((arguments or {}).get("reason") or "").strip()
    if reason:
        logger.info("readonly_sql reason=%s", reason[:200])
    return run_select(sql)


def run_query_scan_tyre_tool(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"tyre_code": arguments}
    args = arguments or {}
    tyre = str(args.get("tyre_code") or "").strip()
    parent = str(args.get("parent_code") or "").strip() or None
    reason = str(args.get("reason") or "").strip()
    if reason:
        logger.info("query_scan_tyre reason=%s tyre=%s parent=%s", reason[:160], tyre, parent)
    return query_scan_tyre_records(tyre, parent_code=parent)


def run_query_warranty_tool(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"license_plate": arguments}
    args = arguments or {}
    plate = str(args.get("license_plate") or "").strip()
    reason = str(args.get("reason") or "").strip()
    if reason:
        logger.info("query_warranty reason=%s plate=%s", reason[:160], plate)
    return query_warranty_by_plate(plate)


def _create_with_retry(client, *, kwargs: dict[str, Any], attempts: int):
    last_exc: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("tool-loop chat failed attempt=%s: %s", attempt + 1, exc)
    raise RuntimeError(f"gpt-sol 工具轮次失败: {last_exc}")


def _execute_tool_calls(
    tool_calls: list[Any],
    *,
    working: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    on_meta: MetaCallback | None,
    assistant_content: str = "",
    default_tyre: str | None = None,
    default_parent: str | None = None,
    default_plate: str | None = None,
) -> None:
    normalized_args: list[str] = []
    for tc in tool_calls:
        raw = tc.function.arguments or "{}"
        name = tc.function.name
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except json.JSONDecodeError:
            args = {}
        if name == "query_scan_tyre_records":
            if not str(args.get("tyre_code") or "").strip() and default_tyre:
                args["tyre_code"] = default_tyre
            if not str(args.get("parent_code") or "").strip() and default_parent:
                args["parent_code"] = default_parent
            raw = json.dumps(args, ensure_ascii=False)
        elif name == "query_warranty_by_plate":
            if not str(args.get("license_plate") or "").strip() and default_plate:
                args["license_plate"] = default_plate
            raw = json.dumps(args, ensure_ascii=False)
        normalized_args.append(raw)

    working.append(
        {
            "role": "assistant",
            "content": assistant_content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": normalized_args[i],
                    },
                }
                for i, tc in enumerate(tool_calls)
            ],
        }
    )

    for i, tc in enumerate(tool_calls):
        name = tc.function.name
        raw_args = normalized_args[i]
        if name == "query_scan_tyre_records":
            stage_msg = "正在按分表查询扫码胎号明细…"
        elif name == "query_warranty_by_plate":
            stage_msg = "正在按车牌查询质保卡…"
        else:
            stage_msg = "正在查询业务只读库核对数据…"
        if on_meta:
            on_meta(
                {
                    "type": "meta",
                    "stage": "sql_query",
                    "message": stage_msg,
                    "tool": name,
                }
            )
        try:
            if name == "query_scan_tyre_records":
                payload = run_query_scan_tyre_tool(raw_args)
            elif name == "query_warranty_by_plate":
                payload = run_query_warranty_tool(raw_args)
            elif name == "run_readonly_sql":
                payload = run_readonly_sql_tool(raw_args)
            else:
                payload = {"ok": False, "error": f"未知工具: {name}"}
            if "error" not in payload:
                payload.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001
            payload = {"ok": False, "error": str(exc)}

        ok = bool(payload.get("ok")) and not payload.get("error")
        traces.append(
            {
                "tool": name,
                "arguments": raw_args[:500],
                "ok": ok,
                "row_count": payload.get("row_count"),
                "error": payload.get("error"),
                "route": (payload.get("route") if isinstance(payload, dict) else None),
            }
        )
        if on_meta:
            on_meta(
                {
                    "type": "meta",
                    "stage": "sql_done",
                    "message": (
                        f"查库完成，返回 {payload.get('row_count', 0)} 行"
                        if ok
                        else f"查库失败：{payload.get('error')}"
                    ),
                    "row_count": payload.get("row_count"),
                    "tool": name,
                }
            )
        working.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _tool_result_payload(payload),
            }
        )


def chat_with_optional_tools(
    messages: list[dict[str, Any]],
    *,
    model_key: str = "gpt-sol",
    settings=None,
    on_meta: MetaCallback | None = None,
    require_sql: bool = False,
    current_user_text: str = "",
    current_image_description: str = "",
) -> dict[str, Any]:
    """非流式工具循环；require_sql 时按本轮意图强制对应工具。"""
    cfg = settings or get_settings()
    client = llm_service._client_for_provider(  # noqa: SLF001
        llm_service.CHAT_MODEL_CATALOG[model_key]["provider"]
    )
    model_name = llm_service._resolve_model_name(model_key, cfg)  # noqa: SLF001
    working: list[dict[str, Any]] = [dict(m) for m in messages]
    tools = (
        [
            QUERY_WARRANTY_BY_PLATE_TOOL,
            QUERY_SCAN_TYRE_TOOL,
            READONLY_SQL_TOOL,
        ]
        if biz_readonly_enabled()
        else []
    )
    max_rounds = max(0, int(cfg.agent_max_tool_rounds))
    traces: list[dict[str, Any]] = []
    attempts = max(1, int(cfg.llm_max_attempts))
    must_sql = bool(require_sql) and bool(tools)
    nudged = False

    # 关键：只从本轮用户问句提取实体，避免历史串题
    turn_text = (current_user_text or "").strip() or _last_user_text(working)
    intent: BizIntent = detect_biz_intent(turn_text, current_image_description)
    inferred_tyre = extract_tyre_code(turn_text, current_image_description)
    inferred_parent = extract_parent_code(turn_text, current_image_description)
    inferred_plate = extract_license_plate(turn_text, current_image_description)

    if not tools:
        result = llm_service.chat_messages_with_retry(
            [{"role": m["role"], "content": m.get("content") or ""} for m in working],
            model_key=model_key,
            settings=cfg,
        )
        return {
            "reply": result.get("reply") or "",
            "model": result.get("model"),
            "key": result.get("key"),
            "tool_traces": [],
            "used_tools": False,
            "require_sql": require_sql,
            "intent": intent,
        }

    for round_idx in range(max_rounds + 1):
        force_final = round_idx >= max_rounds
        need_force_sql = must_sql and not traces and not force_final
        if need_force_sql:
            if intent == "warranty_plate":
                tool_choice: Any = {
                    "type": "function",
                    "function": {"name": "query_warranty_by_plate"},
                }
            elif intent == "scan_tyre" and inferred_tyre:
                tool_choice = {
                    "type": "function",
                    "function": {"name": "query_scan_tyre_records"},
                }
            else:
                tool_choice = {
                    "type": "function",
                    "function": {"name": "run_readonly_sql"},
                }
        elif force_final:
            tool_choice = "none"
        else:
            tool_choice = "auto"

        if need_force_sql and round_idx == 0 and not any(m.get("role") == "tool" for m in working):
            if intent == "warranty_plate" and inferred_plate:
                working.append(
                    {
                        "role": "user",
                        "content": (
                            f"请立即调用 query_warranty_by_plate，"
                            f"参数 license_plate={inferred_plate}。"
                            "不要使用上一轮对话的胎号，不要查扫码流水。"
                        ),
                    }
                )
            elif intent == "scan_tyre" and inferred_tyre:
                hint = (
                    f"请立即调用 query_scan_tyre_records，参数 tyre_code={inferred_tyre}"
                )
                if inferred_parent:
                    hint += f", parent_code={inferred_parent}"
                hint += "。只用本轮胎号。"
                working.append({"role": "user", "content": hint})

        response = _create_with_retry(
            client,
            kwargs={
                "model": model_name,
                "temperature": cfg.openai_temperature,
                "messages": working,
                "tools": tools,
                "tool_choice": tool_choice,
            },
            attempts=attempts,
        )
        msg = response.choices[0].message
        tool_calls = [] if force_final else list(msg.tool_calls or [])

        if not tool_calls:
            if must_sql and not traces and not force_final and not nudged:
                nudged = True
                working.append({"role": "user", "content": _FORCE_SQL_NUDGE})
                if on_meta:
                    on_meta(
                        {
                            "type": "meta",
                            "stage": "sql_query",
                            "message": "要求先查业务只读库核对…",
                        }
                    )
                continue
            return {
                "reply": (msg.content or "").strip(),
                "model": model_name,
                "key": model_key,
                "tool_traces": traces,
                "used_tools": bool(traces),
                "require_sql": require_sql,
                "intent": intent,
            }

        _execute_tool_calls(
            tool_calls,
            working=working,
            traces=traces,
            on_meta=on_meta,
            assistant_content=msg.content or "",
            default_tyre=inferred_tyre if intent == "scan_tyre" else None,
            default_parent=inferred_parent if intent == "scan_tyre" else None,
            default_plate=inferred_plate if intent == "warranty_plate" else None,
        )

    return {
        "reply": "",
        "model": model_name,
        "key": model_key,
        "tool_traces": traces,
        "used_tools": bool(traces),
        "require_sql": require_sql,
        "intent": intent,
    }


def iter_answer_with_tools(
    messages: list[dict[str, Any]],
    *,
    model_key: str = "gpt-sol",
    settings=None,
    require_sql: bool = False,
    current_user_text: str = "",
    current_image_description: str = "",
) -> Iterator[dict[str, Any]]:
    """供 SSE：先产出查库 meta，再产出最终 end。"""
    collected: list[dict[str, Any]] = []

    def collect(ev: dict[str, Any]) -> None:
        collected.append(ev)

    result = chat_with_optional_tools(
        messages,
        model_key=model_key,
        settings=settings,
        on_meta=collect,
        require_sql=require_sql,
        current_user_text=current_user_text,
        current_image_description=current_image_description,
    )
    for ev in collected:
        yield ev
    yield {
        "type": "end",
        "key": result.get("key") or model_key,
        "model": result.get("model"),
        "reply": result.get("reply") or "",
        "tool_traces": result.get("tool_traces") or [],
        "used_tools": result.get("used_tools", False),
        "require_sql": result.get("require_sql", require_sql),
        "intent": result.get("intent"),
    }
