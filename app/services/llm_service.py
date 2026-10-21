"""多模型聊天：仅公司模型 gpt-sol（失败按次数重试，不切换千问）。

带图链路：PaddleOCR 识字 → gpt-sol 综合回答。
"""

from __future__ import annotations

import base64
import logging
import re
import time
from collections.abc import Iterator
from typing import Any

import httpx

from app.common.text import clip_text, summarize_messages
from app.core.config import Settings, get_settings
from app.infra.llm import (
    client_for_provider,
    get_ollama_openai_client,
    get_openai_client,
)


logger = logging.getLogger(__name__)

# 兼容旧调用方
_client_for_provider = client_for_provider

# 逻辑模型 key -> 配置（provider + 真实模型名）
CHAT_MODEL_CATALOG: dict[str, dict[str, str]] = {
    "gpt-sol": {
        "provider": "openai",
        "label": "GPT Sol（网关）",
        "description": "OpenAI 兼容网关 gpt-5.6-sol",
    },
    "qwen2.5-7b": {
        "provider": "ollama",
        "label": "Qwen2.5 7B（本地）",
        "description": "Ollama qwen2.5:7b-instruct-q4_K_M",
    },
}

_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL | re.IGNORECASE)


def _resolve_model_name(key: str, settings: Settings) -> str:
    if key == "gpt-sol":
        return settings.openai_model
    if key == "qwen2.5-7b":
        return settings.ollama_chat_model_qwen
    raise ValueError(f"未知模型 key: {key}，可选: {', '.join(CHAT_MODEL_CATALOG)}")


def list_chat_models(*, settings: Settings | None = None) -> list[dict[str, Any]]:
    cfg = settings or get_settings()
    default_key = cfg.default_chat_model
    items: list[dict[str, Any]] = []
    for key, meta in CHAT_MODEL_CATALOG.items():
        items.append(
            {
                "key": key,
                "provider": meta["provider"],
                "label": meta["label"],
                "description": meta["description"],
                "model": _resolve_model_name(key, cfg),
                "is_default": key == default_key,
            }
        )
    return items


def normalize_image_input(image: str) -> tuple[str, str]:
    """返回 (mime, base64_raw)。支持 data URL 或纯 base64。"""
    text = (image or "").strip()
    if not text:
        raise ValueError("图片内容为空")
    match = _DATA_URL_RE.match(text)
    if match:
        return match.group(1).lower(), match.group(2).replace("\n", "")
    # 纯 base64
    raw = text.replace("\n", "")
    try:
        base64.b64decode(raw, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("图片须为 data:image/...;base64,... 或合法 base64") from exc
    return "image/png", raw


# 兼容旧私有名
_normalize_image_input = normalize_image_input


def _chat_text(
    *,
    model_key: str,
    user_message: str,
    system_message: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    if model_key not in CHAT_MODEL_CATALOG:
        raise ValueError(f"未知模型 key: {model_key}")
    provider = CHAT_MODEL_CATALOG[model_key]["provider"]
    model_name = _resolve_model_name(model_key, cfg)
    client = _client_for_provider(provider)

    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model_name,
        temperature=cfg.openai_temperature if temperature is None else temperature,
        messages=messages,
    )
    content = response.choices[0].message.content
    return {
        "key": model_key,
        "provider": provider,
        "model": model_name,
        "reply": content or "",
    }


def describe_images_ocr(
    images: list[str],
    *,
    question_hint: str = "",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """用 PaddleOCR 识别图片文字。"""
    from app.services import ocr_service

    _ = settings  # 预留阈值阈值等配置
    hint = (question_hint or "").strip()
    logger.info(
        "OCR request images=%s hint=%r",
        len(images or []),
        clip_text(hint, limit=80),
    )
    result = ocr_service.ocr_images(images)
    if hint:
        # 仅附加到返回结构，不改 OCR 原文主体
        result = {**result, "question_hint": hint}
    logger.info(
        "OCR response preview=%r",
        clip_text(result.get("reply"), limit=100),
    )
    return result


def chat(
    user_message: str,
    *,
    system_message: str | None = None,
    model_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """调用指定聊天模型（纯文本）。"""
    cfg = settings or get_settings()
    key = (model_key or cfg.default_chat_model or "gpt-sol").strip()
    if key not in CHAT_MODEL_CATALOG:
        if model and model == cfg.openai_model:
            key = "gpt-sol"
        else:
            raise ValueError(f"未知模型 key: {key}，可选: {', '.join(CHAT_MODEL_CATALOG)}")
    return _chat_text(
        model_key=key,
        user_message=user_message,
        system_message=system_message,
        temperature=temperature,
        settings=cfg,
    )


def chat_messages(
    messages: list[dict[str, str]],
    *,
    model_key: str = "gpt-sol",
    temperature: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """多轮 messages 聊天（已含 system/user/assistant）。"""
    cfg = settings or get_settings()
    if model_key not in CHAT_MODEL_CATALOG:
        raise ValueError(f"未知模型 key: {model_key}")
    if not messages:
        raise ValueError("messages 不能为空")
    provider = CHAT_MODEL_CATALOG[model_key]["provider"]
    model_name = _resolve_model_name(model_key, cfg)
    client = _client_for_provider(provider)
    logger.info(
        "LLM call start key=%s model=%s %s",
        model_key,
        model_name,
        summarize_messages(messages),
    )
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        temperature=cfg.openai_temperature if temperature is None else temperature,
        messages=messages,
    )
    content = response.choices[0].message.content or ""
    logger.info(
        "LLM call done key=%s model=%s ms=%s preview=%r",
        model_key,
        model_name,
        int((time.perf_counter() - t0) * 1000),
        clip_text(content, limit=100),
    )
    return {
        "key": model_key,
        "provider": provider,
        "model": model_name,
        "reply": content,
    }


def chat_messages_with_retry(
    messages: list[dict[str, str]],
    *,
    model_key: str = "gpt-sol",
    temperature: float | None = None,
    settings: Settings | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    """公司模型同步调用，失败按次数重试。"""
    cfg = settings or get_settings()
    attempts = max(1, int(max_attempts if max_attempts is not None else cfg.llm_max_attempts))
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            if i > 0:
                logger.info("LLM retry key=%s attempt=%s/%s", model_key, i + 1, attempts)
            result = chat_messages(
                messages,
                model_key=model_key,
                temperature=temperature,
                settings=cfg,
            )
            if i > 0:
                result = {**result, "retries": i}
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "%s 调用失败 attempt=%s/%s: %s",
                model_key,
                i + 1,
                attempts,
                exc,
            )
            if i + 1 < attempts:
                time.sleep(min(1.5 * (i + 1), 4.0))
    assert last_exc is not None
    raise last_exc


def iter_chat_messages_stream_with_retry(
    *,
    messages: list[dict[str, str]],
    model_key: str = "gpt-sol",
    temperature: float | None = None,
    settings: Settings | None = None,
    max_attempts: int | None = None,
) -> Iterator[dict[str, Any]]:
    """流式调用公司模型；整轮失败则重试（先缓冲，成功后再吐出，避免半截重复）。

    额外事件：meta stage=retry
    """
    cfg = settings or get_settings()
    attempts = max(1, int(max_attempts if max_attempts is not None else cfg.llm_max_attempts))
    last_error = ""

    for i in range(attempts):
        if i > 0:
            yield {
                "type": "meta",
                "stage": "retry",
                "key": model_key,
                "attempt": i + 1,
                "max_attempts": attempts,
                "message": f"公司模型失败，正在重试（{i + 1}/{attempts}）…",
            }
            time.sleep(min(1.5 * i, 4.0))

        buffered: list[dict[str, Any]] = []
        got_end = False
        full_reply = ""
        err_msg = ""
        for ev in iter_chat_messages_stream(
            messages=messages,
            model_key=model_key,
            temperature=temperature,
            settings=cfg,
        ):
            if ev["type"] == "error":
                err_msg = str(ev.get("message") or "调用失败")
                break
            if ev["type"] == "end":
                got_end = True
                full_reply = ev.get("reply") or ""
            buffered.append(ev)

        if got_end and full_reply.strip():
            for ev in buffered:
                yield ev
            return

        if got_end and not full_reply.strip():
            err_msg = err_msg or "模型返回空内容"
        last_error = err_msg or last_error or "调用失败"
        logger.warning(
            "%s 流式失败 attempt=%s/%s: %s",
            model_key,
            i + 1,
            attempts,
            last_error,
        )

    yield {
        "type": "error",
        "key": model_key,
        "message": last_error or f"{model_key} 重试后仍失败",
    }


def chat_with_pipeline(
    user_message: str,
    *,
    images: list[str] | None = None,
    system_message: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """带图/纯文本编排：仅 gpt-sol，失败重试。"""
    cfg = settings or get_settings()
    system = system_message or "你是 diagnose-agent 助手，用简洁中文回答。"
    image_list = [x for x in (images or []) if (x or "").strip()]
    steps: list[dict[str, Any]] = []
    errors: list[str] = []

    answer_prompt = user_message
    image_desc = ""

    if image_list:
        try:
            vision = describe_images_ocr(
                image_list,
                question_hint=user_message,
                settings=cfg,
            )
            image_desc = vision["reply"]
            steps.append(
                {
                    "step": "ocr",
                    "ok": True,
                    "key": vision["key"],
                    "model": vision["model"],
                    "reply": image_desc,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("PaddleOCR 调用失败: %s", exc)
            errors.append(f"paddleocr: {exc}")
            steps.append({"step": "ocr", "ok": False, "error": str(exc)})

        answer_prompt = (
            "下面是图片 OCR 识别出的文字与用户原始问题，请综合回答用户问题。"
            "请以 OCR 文字为准，不要编造图中不存在的内容。\n\n"
            f"【图片OCR文字】\n{image_desc or '（OCR失败，仅根据文字问题作答）'}\n\n"
            f"【用户问题】\n{user_message.strip() or '（用户未提供文字，请根据OCR文字总结）'}"
        )

    openai_messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": answer_prompt},
    ]
    try:
        sol = chat_messages_with_retry(
            openai_messages,
            model_key="gpt-sol",
            temperature=temperature,
            settings=cfg,
        )
        steps.append(
            {"step": "answer", "ok": True, "key": sol["key"], "model": sol["model"]}
        )
        return {
            "key": sol["key"],
            "provider": sol["provider"],
            "model": sol["model"],
            "reply": sol["reply"],
            "pipeline": "paddleocr->gpt-sol" if image_list else "gpt-sol",
            "image_description": image_desc,
            "fallback": False,
            "retries": sol.get("retries", 0),
            "steps": steps,
            "errors": errors,
        }
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gpt-sol: {exc}")
        steps.append({"step": "answer", "ok": False, "error": str(exc)})
        raise RuntimeError("gpt-sol 重试后仍失败: " + " | ".join(errors)) from exc


def iter_chat_text_stream(
    *,
    model_key: str,
    user_message: str,
    system_message: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> Iterator[dict[str, Any]]:
    """流式纯文本聊天。yield: start / delta / end / error。"""
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    yield from iter_chat_messages_stream(
        messages=messages,
        model_key=model_key,
        temperature=temperature,
        settings=settings,
    )


def iter_chat_messages_stream(
    *,
    messages: list[dict[str, str]],
    model_key: str,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> Iterator[dict[str, Any]]:
    """多轮 messages 流式聊天。yield: start / delta / end / error。"""
    cfg = settings or get_settings()
    if model_key not in CHAT_MODEL_CATALOG:
        yield {"type": "error", "message": f"未知模型 key: {model_key}"}
        return
    if not messages:
        yield {"type": "error", "message": "messages 不能为空"}
        return

    provider = CHAT_MODEL_CATALOG[model_key]["provider"]
    model_name = _resolve_model_name(model_key, cfg)
    client = _client_for_provider(provider)

    logger.info(
        "LLM stream start key=%s model=%s %s",
        model_key,
        model_name,
        summarize_messages(messages),
    )
    yield {
        "type": "start",
        "key": model_key,
        "provider": provider,
        "model": model_name,
    }
    t0 = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model_name,
            temperature=cfg.openai_temperature if temperature is None else temperature,
            messages=messages,
            stream=True,
        )
        full = []
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content or ""
            if piece:
                full.append(piece)
                yield {"type": "delta", "text": piece}
        reply = "".join(full)
        logger.info(
            "LLM stream done key=%s model=%s ms=%s preview=%r",
            model_key,
            model_name,
            int((time.perf_counter() - t0) * 1000),
            clip_text(reply, limit=100),
        )
        yield {
            "type": "end",
            "key": model_key,
            "provider": provider,
            "model": model_name,
            "reply": reply,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("流式调用失败 model=%s: %s", model_key, exc)
        yield {"type": "error", "key": model_key, "message": str(exc)}



def iter_pipeline_stream(
    user_message: str,
    *,
    images: list[str] | None = None,
    system_message: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> Iterator[dict[str, Any]]:
    """编排流：OCR(非流) → answer(stream) → 必要时 fallback stream。

    事件：meta / delta / done / error
    """
    cfg = settings or get_settings()
    system = system_message or "你是 diagnose-agent 助手，用简洁中文回答。"
    image_list = [x for x in (images or []) if (x or "").strip()]
    errors: list[str] = []
    image_desc = ""

    if image_list:
        yield {"type": "meta", "stage": "vision", "message": "正在 OCR 识别图片文字…"}
        try:
            vision = describe_images_ocr(
                image_list,
                question_hint=user_message,
                settings=cfg,
            )
            image_desc = vision["reply"]
            yield {
                "type": "meta",
                "stage": "vision_done",
                "key": "paddleocr",
                "model": vision["model"],
                "image_description": image_desc,
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"paddleocr: {exc}")
            yield {
                "type": "meta",
                "stage": "vision_failed",
                "error": str(exc),
            }

        answer_prompt = (
            "下面是图片 OCR 识别出的文字与用户原始问题，请综合回答用户问题。"
            "请以 OCR 文字为准，不要编造图中不存在的内容。\n\n"
            f"【图片OCR文字】\n{image_desc or '（OCR失败，仅根据文字问题作答）'}\n\n"
            f"【用户问题】\n{user_message.strip() or '（用户未提供文字，请根据OCR文字总结）'}"
        )
    else:
        answer_prompt = user_message

    yield {"type": "meta", "stage": "answer", "key": "gpt-sol", "message": "正在生成回答…"}
    full_reply = ""
    answer_key = "gpt-sol"
    answer_model = _resolve_model_name("gpt-sol", cfg)
    pipeline = "gpt-sol" if not image_list else "paddleocr->gpt-sol"
    openai_messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": answer_prompt},
    ]

    for ev in iter_chat_messages_stream_with_retry(
        messages=openai_messages,
        model_key="gpt-sol",
        temperature=temperature,
        settings=cfg,
    ):
        if ev["type"] == "meta":
            yield ev
        elif ev["type"] == "start":
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
            full_reply = ev.get("reply") or ""
            answer_key = "gpt-sol"
        elif ev["type"] == "error":
            errors.append(f"gpt-sol: {ev.get('message')}")
            yield {"type": "error", "message": str(ev.get("message") or "gpt-sol 失败")}
            return

    if not full_reply and errors:
        yield {"type": "error", "message": " | ".join(errors)}
        return

    yield {
        "type": "done",
        "key": answer_key,
        "model": answer_model,
        "provider": CHAT_MODEL_CATALOG[answer_key]["provider"],
        "reply": full_reply,
        "pipeline": pipeline,
        "image_description": image_desc,
        "fallback": False,
        "errors": errors,
    }
