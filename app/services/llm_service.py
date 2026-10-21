"""多模型聊天：OpenAI 网关 gpt-sol（优先）+ 本地 Ollama qwen2.5-7b（兜底）。

带图链路：PaddleOCR 识字 → gpt-sol 综合回答；失败则 qwen2.5-7b。
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import httpx
from openai import OpenAI

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

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


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY，请写入 .env")
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.normalized_openai_base_url(),
    )


@lru_cache
def get_ollama_openai_client() -> OpenAI:
    """Ollama 的 OpenAI 兼容接口（/v1）。"""
    settings = get_settings()
    base = settings.ollama_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return OpenAI(api_key="ollama", base_url=base)


def _client_for_provider(provider: str) -> OpenAI:
    if provider == "openai":
        return get_openai_client()
    if provider == "ollama":
        return get_ollama_openai_client()
    raise ValueError(f"不支持的 provider: {provider}")


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
    result = ocr_service.ocr_images(images)
    hint = (question_hint or "").strip()
    if hint:
        # 仅附加到返回结构，不改 OCR 原文主体
        result = {**result, "question_hint": hint}
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
    model_key: str = "qwen2.5-7b",
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


def chat_with_pipeline(
    user_message: str,
    *,
    images: list[str] | None = None,
    system_message: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """带图/纯文本编排。

    有图片时：
      1) PaddleOCR 识别文字
      2) OCR 文本 + 原始问题 → gpt-sol（公司模型）
      3) 失败 → qwen2.5-7b 兜底

    无图片时：优先 gpt-sol，失败再 qwen2.5-7b。
    """
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

    try:
        sol = _chat_text(
            model_key="gpt-sol",
            user_message=answer_prompt,
            system_message=system,
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
            "steps": steps,
            "errors": errors,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpt-sol 调用失败: %s", exc)
        errors.append(f"gpt-sol: {exc}")
        steps.append({"step": "answer", "ok": False, "error": str(exc)})

    try:
        qwen = _chat_text(
            model_key="qwen2.5-7b",
            user_message=answer_prompt,
            system_message=system,
            temperature=temperature,
            settings=cfg,
        )
        steps.append(
            {"step": "fallback", "ok": True, "key": qwen["key"], "model": qwen["model"]}
        )
        return {
            "key": qwen["key"],
            "provider": qwen["provider"],
            "model": qwen["model"],
            "reply": qwen["reply"],
            "pipeline": (
                "fallback:qwen2.5-7b(after-gpt-sol-failed)"
                if image_list
                else "fallback:qwen2.5-7b"
            ),
            "image_description": image_desc,
            "fallback": True,
            "steps": steps,
            "errors": errors,
        }
    except Exception as exc:  # noqa: BLE001
        errors.append(f"qwen2.5-7b: {exc}")
        steps.append({"step": "fallback", "ok": False, "error": str(exc)})
        raise RuntimeError("gpt-sol 与 qwen2.5-7b 均调用失败: " + " | ".join(errors)) from exc


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

    yield {
        "type": "start",
        "key": model_key,
        "provider": provider,
        "model": model_name,
    }
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
        yield {
            "type": "end",
            "key": model_key,
            "provider": provider,
            "model": model_name,
            "reply": "".join(full),
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
    primary_ok = False
    full_reply = ""
    answer_key = "gpt-sol"
    answer_model = _resolve_model_name("gpt-sol", cfg)
    fallback = False
    pipeline = "gpt-sol" if not image_list else "paddleocr->gpt-sol"

    for ev in iter_chat_text_stream(
        model_key="gpt-sol",
        user_message=answer_prompt,
        system_message=system,
        temperature=temperature,
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
            primary_ok = True
            full_reply = ev.get("reply") or ""
            answer_key = "gpt-sol"
        elif ev["type"] == "error":
            errors.append(f"gpt-sol: {ev.get('message')}")

    if not primary_ok:
        pipeline = (
            "fallback:qwen2.5-7b(after-gpt-sol-failed)"
            if image_list
            else "fallback:qwen2.5-7b"
        )
        yield {
            "type": "meta",
            "stage": "fallback",
            "key": "qwen2.5-7b",
            "message": "公司模型不可用，切换本地千问…",
        }
        full_reply = ""
        for ev in iter_chat_text_stream(
            model_key="qwen2.5-7b",
            user_message=answer_prompt,
            system_message=system,
            temperature=temperature,
            settings=cfg,
        ):
            if ev["type"] == "start":
                answer_key = "qwen2.5-7b"
                answer_model = ev.get("model") or _resolve_model_name("qwen2.5-7b", cfg)
                fallback = True
                yield {
                    "type": "meta",
                    "stage": "answer_start",
                    "key": "qwen2.5-7b",
                    "model": answer_model,
                    "fallback": True,
                }
            elif ev["type"] == "delta":
                yield {"type": "delta", "text": ev.get("text") or ""}
            elif ev["type"] == "end":
                full_reply = ev.get("reply") or ""
                answer_key = "qwen2.5-7b"
                fallback = True
            elif ev["type"] == "error":
                errors.append(f"qwen2.5-7b: {ev.get('message')}")
                yield {
                    "type": "error",
                    "message": str(ev.get("message") or "qwen2.5-7b 失败"),
                }
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
        "fallback": fallback,
        "errors": errors,
    }
