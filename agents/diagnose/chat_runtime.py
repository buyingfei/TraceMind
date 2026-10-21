"""带 Postgres 记忆的对话运行时：OCR → 检索 → 多轮 LLM → 写回 checkpoint。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agents.diagnose.graph import get_chat_graph
from agents.diagnose.biz_verify import needs_biz_sql_verify
from agents.diagnose.nodes.chat_answer import (
    build_human_content,
    build_openai_messages_for_turn,
    messages_for_llm,
    resolve_answer_audience,
    sanitize_reply_for_audience,
)
from agents.diagnose.nodes.retrieve_code import run_retrieve
from agents.diagnose.tool_loop import iter_answer_with_tools
from agents.tools.readonly_sql import biz_readonly_enabled
from app.core.config import get_settings
from app.core.logging import clip_text
from app.services import llm_service
from app.services.checkpoint_service import thread_config

logger = logging.getLogger(__name__)


def _snapshot_messages(conversation_id: int) -> list[Any]:
    graph = get_chat_graph()
    config = thread_config(conversation_id)
    snap = graph.get_state(config)
    values = (snap.values if snap else None) or {}
    return list(values.get("messages") or [])


def _bootstrap_from_mysql_if_empty(
    conversation_id: int,
    existing: list[Any],
) -> list[Any]:
    """checkpoint 为空时，用 MySQL 历史重建（不含本轮尚未入库完成的最后一条 user）。"""
    if existing:
        return existing
    try:
        from app.services.chat_service import list_messages

        rows = list_messages(conversation_id)
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("bootstrap MySQL 失败 conversation_id=%s: %s", conversation_id, exc)
        return []

    rebuilt: list[Any] = []
    trim = rows
    if trim and trim[-1].get("role") == "user":
        trim = trim[:-1]
    for row in trim:
        role = row.get("role")
        content = (row.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            rebuilt.append(HumanMessage(content=content))
        elif role == "assistant":
            rebuilt.append(AIMessage(content=content))
    if rebuilt:
        logger.info(
            "bootstrap checkpoint from MySQL conversation_id=%s count=%s",
            conversation_id,
            len(rebuilt),
        )
        graph = get_chat_graph()
        graph.update_state(
            thread_config(conversation_id),
            {"messages": rebuilt},
        )
    return rebuilt


def _persist_turn(
    conversation_id: int,
    human_text: str,
    ai_text: str,
    *,
    code_hits: list[dict[str, Any]] | None = None,
) -> None:
    """把本轮 Human+AI 与检索摘要写入 checkpoint。"""
    graph = get_chat_graph()
    config = thread_config(conversation_id)
    payload: dict[str, Any] = {
        "messages": [
            HumanMessage(content=human_text),
            AIMessage(content=ai_text),
        ],
        "reply": ai_text,
    }
    if code_hits is not None:
        payload["code_hits"] = code_hits
    graph.update_state(config, payload)


def _hit_summaries(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": h.get("path"),
            "symbol": h.get("symbol"),
            "start_line": h.get("start_line"),
            "end_line": h.get("end_line"),
            "score": h.get("score"),
        }
        for h in (hits or [])[:8]
    ]


def run_chat_turn(
    conversation_id: int,
    *,
    user_text: str,
    images: list[str] | None = None,
) -> dict[str, Any]:
    """同步一轮：OCR → retrieve → graph.invoke → 返回结果。"""
    cfg = get_settings()
    image_list = [x for x in (images or []) if (x or "").strip()]
    image_desc = ""
    errors: list[str] = []

    if image_list:
        try:
            vision = llm_service.describe_images_ocr(
                image_list,
                question_hint=user_text,
                settings=cfg,
            )
            image_desc = vision.get("reply") or ""
        except Exception as exc:  # noqa: BLE001
            errors.append(f"paddleocr: {exc}")

    human = build_human_content(user_text, image_description=image_desc)
    existing = _snapshot_messages(conversation_id)
    _bootstrap_from_mysql_if_empty(conversation_id, existing)

    graph = get_chat_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=human)],
            "user_text": user_text,
            "image_description": image_desc,
            "errors": errors,
        },
        config=thread_config(conversation_id),
    )
    code_hits = list(result.get("code_hits") or [])
    return {
        "reply": result.get("reply") or "",
        "key": result.get("model_key") or "gpt-sol",
        "model": llm_service._resolve_model_name(  # noqa: SLF001
            result.get("model_key") or "gpt-sol", cfg
        ),
        "provider": llm_service.CHAT_MODEL_CATALOG.get(
            result.get("model_key") or "gpt-sol", {}
        ).get("provider", "openai"),
        "pipeline": result.get("pipeline") or "agent:retrieve+memory",
        "image_description": image_desc,
        "fallback": False,
        "errors": result.get("errors") or errors,
        "code_hits": _hit_summaries(code_hits),
        "memory": True,
    }


def iter_chat_turn_stream(
    conversation_id: int,
    *,
    user_text: str,
    images: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """流式一轮：OCR → retrieve meta → 带历史+证据流式生成 → 写 checkpoint。"""
    cfg = get_settings()
    image_list = [x for x in (images or []) if (x or "").strip()]
    image_desc = ""
    errors: list[str] = []
    logger.info(
        "chat turn start conversation_id=%s images=%s user=%r",
        conversation_id,
        len(image_list),
        clip_text(user_text, limit=100),
    )

    if image_list:
        yield {"type": "meta", "stage": "vision", "message": "正在 OCR 识别图片文字…"}
        try:
            vision = llm_service.describe_images_ocr(
                image_list,
                question_hint=user_text,
                settings=cfg,
            )
            image_desc = vision.get("reply") or ""
            yield {
                "type": "meta",
                "stage": "vision_done",
                "key": "paddleocr",
                "model": vision.get("model"),
                "image_description": image_desc,
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"paddleocr: {exc}")
            logger.warning("chat turn OCR failed conversation_id=%s: %s", conversation_id, exc)
            yield {"type": "meta", "stage": "vision_failed", "error": str(exc)}

    human = build_human_content(user_text, image_description=image_desc)
    existing = _snapshot_messages(conversation_id)
    history_msgs = _bootstrap_from_mysql_if_empty(conversation_id, existing)
    if not history_msgs:
        history_msgs = _snapshot_messages(conversation_id)

    yield {
        "type": "meta",
        "stage": "retrieve",
        "message": "正在检索代码向量库…",
    }
    retrieve = run_retrieve(user_text=user_text, image_description=image_desc)
    code_hits = list(retrieve.get("code_hits") or [])
    if retrieve.get("retrieve_error"):
        errors.append(f"retrieve: {retrieve['retrieve_error']}")
    path_boost = bool(retrieve.get("path_boost"))
    logger.info(
        "chat retrieve done conversation_id=%s hits=%s path_boost=%s queries=%s",
        conversation_id,
        len(code_hits),
        path_boost,
        len(retrieve.get("retrieve_queries") or []),
    )
    yield {
        "type": "meta",
        "stage": "retrieve_done",
        "message": (
            f"检索完成，命中 {len(code_hits)} 条代码"
            + ("（已补关键路径）" if path_boost else "")
        ),
        "hit_count": len(code_hits),
        "code_hits": _hit_summaries(code_hits),
        "retrieve_queries": retrieve.get("retrieve_queries") or [],
        "index": retrieve.get("index") or {},
        "path_boost": path_boost,
    }

    audience = resolve_answer_audience(cfg)
    sql_ready = biz_readonly_enabled()
    require_sql = needs_biz_sql_verify(user_text, image_desc) and sql_ready
    openai_history = messages_for_llm(history_msgs)
    openai_messages = build_openai_messages_for_turn(
        history_messages=history_msgs,
        human_content=human,
        code_hits=code_hits,
        audience=audience,
        require_sql=require_sql,
        user_text=user_text,
        image_description=image_desc,
    )
    logger.info(
        "chat answer start conversation_id=%s audience=%s require_sql=%s memory=%s",
        conversation_id,
        audience,
        require_sql,
        len(openai_history),
    )

    yield {
        "type": "meta",
        "stage": "answer",
        "key": "gpt-sol",
        "message": (
            "正在生成回答（将先查业务只读库核对）…"
            if require_sql
            else (
                "正在生成回答（必要时会查业务只读库核对）…"
                if sql_ready
                else "正在生成回答…"
            )
        ),
        "memory_messages": len(openai_history),
        "hit_count": len(code_hits),
        "answer_audience": audience,
        "biz_sql_enabled": sql_ready,
        "require_sql": require_sql,
    }

    full_reply = ""
    answer_key = "gpt-sol"
    answer_model = llm_service._resolve_model_name("gpt-sol", cfg)  # noqa: SLF001
    used_tools = False
    pipeline = (
        "agent:ocr->retrieve->gpt-sol+memory"
        if image_list
        else "agent:retrieve->gpt-sol+memory"
    )

    try:
        for ev in iter_answer_with_tools(
            openai_messages,
            model_key="gpt-sol",
            settings=cfg,
            require_sql=require_sql,
            current_user_text=user_text,
            current_image_description=image_desc,
        ):
            if ev.get("type") == "meta":
                yield ev
            elif ev.get("type") == "end":
                full_reply = sanitize_reply_for_audience(
                    ev.get("reply") or "",
                    audience=audience,
                )
                answer_key = ev.get("key") or "gpt-sol"
                answer_model = ev.get("model") or answer_model
                used_tools = bool(ev.get("used_tools"))
                if require_sql and not used_tools:
                    errors.append("require_sql: 模型未成功完成查库核对")
                if full_reply:
                    yield {"type": "delta", "text": full_reply}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gpt-sol: {exc}")
        yield {
            "type": "error",
            "message": str(exc) or "gpt-sol 失败",
        }
        return

    if not full_reply and errors:
        yield {"type": "error", "message": " | ".join(errors)}
        return

    if used_tools:
        pipeline = (
            "agent:ocr->retrieve->sql->gpt-sol+memory"
            if image_list
            else "agent:retrieve->sql->gpt-sol+memory"
        )

    try:
        _persist_turn(
            conversation_id,
            human,
            full_reply,
            code_hits=code_hits,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("写 checkpoint 失败 conversation_id=%s", conversation_id)
        errors.append(f"checkpoint: {exc}")

    logger.info(
        "chat turn done conversation_id=%s pipeline=%s used_tools=%s errors=%s preview=%r",
        conversation_id,
        pipeline,
        used_tools,
        len(errors),
        clip_text(full_reply, limit=100),
    )
    yield {
        "type": "done",
        "key": answer_key,
        "model": answer_model,
        "provider": llm_service.CHAT_MODEL_CATALOG[answer_key]["provider"],
        "reply": full_reply,
        "pipeline": pipeline,
        "image_description": image_desc,
        "fallback": False,
        "errors": errors,
        "memory": True,
        "used_tools": used_tools,
        "code_hits": _hit_summaries(code_hits),
        "hit_count": len(code_hits),
    }
