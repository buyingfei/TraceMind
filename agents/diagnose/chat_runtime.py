"""带 Postgres 记忆的对话运行时：OCR → 多轮 LLM → 写回 checkpoint。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agents.diagnose.graph import get_chat_graph
from agents.diagnose.nodes.chat_answer import (
    _SYSTEM,
    build_human_content,
    messages_for_llm,
)
from app.core.config import get_settings
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
    # 最后一条若是刚写入的 user，本轮会再追加 HumanMessage，这里跳过最后一条 user
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


def _persist_turn(conversation_id: int, human_text: str, ai_text: str) -> None:
    """把本轮 Human+AI 写入 checkpoint（不经过 answer 节点，避免重复生成）。"""
    graph = get_chat_graph()
    config = thread_config(conversation_id)
    graph.update_state(
        config,
        {
            "messages": [
                HumanMessage(content=human_text),
                AIMessage(content=ai_text),
            ],
            "reply": ai_text,
        },
    )


def run_chat_turn(
    conversation_id: int,
    *,
    user_text: str,
    images: list[str] | None = None,
) -> dict[str, Any]:
    """同步一轮：OCR → graph.invoke（带记忆）→ 返回结果。"""
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
            "image_description": image_desc,
            "errors": errors,
        },
        config=thread_config(conversation_id),
    )
    return {
        "reply": result.get("reply") or "",
        "key": result.get("model_key") or "qwen2.5-7b",
        "model": llm_service._resolve_model_name(  # noqa: SLF001
            result.get("model_key") or "qwen2.5-7b", cfg
        ),
        "provider": llm_service.CHAT_MODEL_CATALOG.get(
            result.get("model_key") or "qwen2.5-7b", {}
        ).get("provider", "ollama"),
        "pipeline": result.get("pipeline") or "agent:memory",
        "image_description": image_desc,
        "fallback": (result.get("model_key") == "gpt-sol"),
        "errors": result.get("errors") or errors,
    }


def iter_chat_turn_stream(
    conversation_id: int,
    *,
    user_text: str,
    images: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """流式一轮：OCR meta → 带历史流式生成 → 写 checkpoint。

    事件与 llm_service.iter_pipeline_stream 对齐：meta / delta / done / error
    """
    cfg = get_settings()
    image_list = [x for x in (images or []) if (x or "").strip()]
    image_desc = ""
    errors: list[str] = []

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
            yield {"type": "meta", "stage": "vision_failed", "error": str(exc)}

    human = build_human_content(user_text, image_description=image_desc)
    existing = _snapshot_messages(conversation_id)
    history_msgs = _bootstrap_from_mysql_if_empty(conversation_id, existing)
    if not history_msgs:
        history_msgs = _snapshot_messages(conversation_id)

    openai_history = messages_for_llm(history_msgs)
    openai_messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM},
        *openai_history,
        {"role": "user", "content": human},
    ]

    yield {
        "type": "meta",
        "stage": "answer",
        "key": "qwen2.5-7b",
        "message": "正在生成回答（含会话记忆）…",
        "memory_messages": len(openai_history),
    }

    full_reply = ""
    answer_key = "qwen2.5-7b"
    answer_model = llm_service._resolve_model_name("qwen2.5-7b", cfg)  # noqa: SLF001
    fallback = False
    pipeline = (
        "agent:paddleocr->qwen2.5-7b+memory"
        if image_list
        else "agent:qwen2.5-7b+memory"
    )
    qwen_ok = False

    for ev in llm_service.iter_chat_messages_stream(
        messages=openai_messages,
        model_key="qwen2.5-7b",
        settings=cfg,
    ):
        if ev["type"] == "start":
            answer_model = ev.get("model") or answer_model
            yield {
                "type": "meta",
                "stage": "answer_start",
                "key": ev.get("key"),
                "model": answer_model,
            }
        elif ev["type"] == "delta":
            yield {"type": "delta", "text": ev.get("text") or ""}
        elif ev["type"] == "end":
            qwen_ok = True
            full_reply = ev.get("reply") or ""
            answer_key = "qwen2.5-7b"
        elif ev["type"] == "error":
            errors.append(f"qwen2.5-7b: {ev.get('message')}")

    if not qwen_ok:
        pipeline = "agent:fallback:gpt-sol+memory"
        yield {
            "type": "meta",
            "stage": "fallback",
            "key": "gpt-sol",
            "message": "本地模型不可用，切换 gpt-sol…",
        }
        for ev in llm_service.iter_chat_messages_stream(
            messages=openai_messages,
            model_key="gpt-sol",
            settings=cfg,
        ):
            if ev["type"] == "start":
                answer_key = "gpt-sol"
                answer_model = ev.get("model") or llm_service._resolve_model_name(  # noqa: SLF001
                    "gpt-sol", cfg
                )
                fallback = True
                yield {
                    "type": "meta",
                    "stage": "answer_start",
                    "key": "gpt-sol",
                    "model": answer_model,
                    "fallback": True,
                }
            elif ev["type"] == "delta":
                yield {"type": "delta", "text": ev.get("text") or ""}
            elif ev["type"] == "end":
                full_reply = ev.get("reply") or ""
                answer_key = "gpt-sol"
                fallback = True
            elif ev["type"] == "error":
                errors.append(f"gpt-sol: {ev.get('message')}")
                yield {"type": "error", "message": str(ev.get("message") or "gpt-sol 失败")}
                return

    if not full_reply and errors:
        yield {"type": "error", "message": " | ".join(errors)}
        return

    try:
        _persist_turn(conversation_id, human, full_reply)
    except Exception as exc:  # noqa: BLE001
        logger.exception("写 checkpoint 失败 conversation_id=%s", conversation_id)
        errors.append(f"checkpoint: {exc}")

    yield {
        "type": "done",
        "key": answer_key,
        "model": answer_model,
        "provider": llm_service.CHAT_MODEL_CATALOG[answer_key]["provider"],
        "reply": full_reply,
        "pipeline": pipeline,
        "image_description": image_desc,
        "fallback": fallback,
        "errors": errors,
        "memory": True,
    }
