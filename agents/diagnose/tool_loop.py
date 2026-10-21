"""公司模型 + 工具循环（只读 SQL 核对）。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from agents.tools.readonly_sql import (
    READONLY_SQL_TOOL,
    biz_readonly_enabled,
    run_select,
)
from app.core.config import get_settings
from app.services import llm_service

logger = logging.getLogger(__name__)

MetaCallback = Callable[[dict[str, Any]], None]

_FORCE_SQL_NUDGE = (
    "系统要求：本题必须先调用 run_readonly_sql 查业务只读库核对关键数据"
    "（胎号流水 / is_auto / 门店操作），再给出最终回答。请立即发起查库，不要直接作答。"
)


def _tool_result_payload(data: dict[str, Any], *, max_chars: int = 6000) -> str:
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "…(truncated)"
    return text


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
) -> None:
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
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ],
        }
    )

    for tc in tool_calls:
        name = tc.function.name
        raw_args = tc.function.arguments or "{}"
        if on_meta:
            on_meta(
                {
                    "type": "meta",
                    "stage": "sql_query",
                    "message": "正在查询业务只读库核对数据…",
                    "tool": name,
                }
            )
        try:
            if name != "run_readonly_sql":
                payload: dict[str, Any] = {"ok": False, "error": f"未知工具: {name}"}
            else:
                payload = run_readonly_sql_tool(raw_args)
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
) -> dict[str, Any]:
    """非流式：必要时多轮调用工具；require_sql 时至少查库一轮再作答。"""
    cfg = settings or get_settings()
    client = llm_service._client_for_provider(  # noqa: SLF001
        llm_service.CHAT_MODEL_CATALOG[model_key]["provider"]
    )
    model_name = llm_service._resolve_model_name(model_key, cfg)  # noqa: SLF001
    working: list[dict[str, Any]] = [dict(m) for m in messages]
    tools = [READONLY_SQL_TOOL] if biz_readonly_enabled() else []
    max_rounds = max(0, int(cfg.agent_max_tool_rounds))
    traces: list[dict[str, Any]] = []
    attempts = max(1, int(cfg.llm_max_attempts))
    must_sql = bool(require_sql) and bool(tools)
    nudged = False

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
        }

    for round_idx in range(max_rounds + 1):
        force_final = round_idx >= max_rounds
        need_force_sql = must_sql and not traces and not force_final
        if need_force_sql:
            tool_choice: Any = {
                "type": "function",
                "function": {"name": "run_readonly_sql"},
            }
        elif force_final:
            tool_choice = "none"
        else:
            tool_choice = "auto"

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
            # 强制查库但模型仍未调工具：追加提示再给一轮机会
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
            }

        _execute_tool_calls(
            tool_calls,
            working=working,
            traces=traces,
            on_meta=on_meta,
            assistant_content=msg.content or "",
        )

    return {
        "reply": "",
        "model": model_name,
        "key": model_key,
        "tool_traces": traces,
        "used_tools": bool(traces),
        "require_sql": require_sql,
    }


def iter_answer_with_tools(
    messages: list[dict[str, Any]],
    *,
    model_key: str = "gpt-sol",
    settings=None,
    require_sql: bool = False,
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
    }
