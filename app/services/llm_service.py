"""多模型聊天：OpenAI 网关 gpt-sol + 本地 Ollama（moondream2 / qwen2.5-7b）。

带图链路：moondream2 解析图片 → qwen2.5-7b 综合回答；两者都失败则 gpt-sol 兜底。
"""

from __future__ import annotations

import base64
import logging
import re
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
    "moondream2": {
        "provider": "ollama",
        "label": "Moondream2（本地）",
        "description": "Ollama 视觉小模型，适合看图简述",
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
    if key == "moondream2":
        return settings.ollama_chat_model_moondream
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


def _normalize_image_input(image: str) -> tuple[str, str]:
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


def describe_images_moondream(
    images: list[str],
    *,
    question_hint: str = "",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """用本地 moondream 解析图片（Ollama 原生 /api/chat + images）。"""
    if not images:
        raise ValueError("images 不能为空")
    cfg = settings or get_settings()
    model_name = cfg.ollama_chat_model_moondream
    b64_list: list[str] = []
    for img in images:
        _, raw = _normalize_image_input(img)
        b64_list.append(raw)

    prompt = "请详细描述图片中的内容，包括文字、界面元素、报错信息。用中文回答。"
    if question_hint.strip():
        prompt += f"\n用户问题供参考：{question_hint.strip()}"

    url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model_name,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": b64_list,
            }
        ],
    }
    with httpx.Client(timeout=180.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    reply = ((data.get("message") or {}).get("content")) or ""
    if not reply.strip():
        raise RuntimeError("moondream2 返回空内容")
    return {
        "key": "moondream2",
        "provider": "ollama",
        "model": model_name,
        "reply": reply.strip(),
    }


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
      1) moondream2 解析图片
      2) 解析结果 + 原始问题 → qwen2.5-7b
      3) moondream 与 qwen 都失败 → gpt-sol 兜底

    无图片时：优先 qwen2.5-7b，失败再 gpt-sol。
    """
    cfg = settings or get_settings()
    system = system_message or "你是 diagnose-agent 助手，用简洁中文回答。"
    image_list = [x for x in (images or []) if (x or "").strip()]
    steps: list[dict[str, Any]] = []
    errors: list[str] = []

    # ---------- 有图：moondream → qwen ----------
    if image_list:
        image_desc = ""
        moondream_ok = False
        qwen_ok = False

        try:
            vision = describe_images_moondream(
                image_list,
                question_hint=user_message,
                settings=cfg,
            )
            image_desc = vision["reply"]
            moondream_ok = True
            steps.append(
                {
                    "step": "vision",
                    "ok": True,
                    "key": vision["key"],
                    "model": vision["model"],
                    "reply": image_desc,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("moondream2 调用失败: %s", exc)
            errors.append(f"moondream2: {exc}")
            steps.append({"step": "vision", "ok": False, "error": str(exc)})

        qwen_prompt = (
            "下面是图片解析结果与用户原始问题，请综合回答用户问题。\n\n"
            f"【图片解析】\n{image_desc or '（图片解析失败，仅根据文字问题作答）'}\n\n"
            f"【用户问题】\n{user_message.strip() or '（用户未提供文字，请根据图片解析作答）'}"
        )
        try:
            qwen = _chat_text(
                model_key="qwen2.5-7b",
                user_message=qwen_prompt,
                system_message=system,
                temperature=temperature,
                settings=cfg,
            )
            qwen_ok = True
            steps.append(
                {
                    "step": "answer",
                    "ok": True,
                    "key": qwen["key"],
                    "model": qwen["model"],
                }
            )
            return {
                "key": qwen["key"],
                "provider": qwen["provider"],
                "model": qwen["model"],
                "reply": qwen["reply"],
                "pipeline": "moondream2->qwen2.5-7b",
                "image_description": image_desc,
                "fallback": False,
                "steps": steps,
                "errors": errors,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("qwen2.5-7b 调用失败: %s", exc)
            errors.append(f"qwen2.5-7b: {exc}")
            steps.append({"step": "answer", "ok": False, "error": str(exc)})

        # 本地两步都失败 → gpt-sol；若仅 qwen 失败但已有图片解析，也用 gpt-sol 带着解析兜底出最终答
        both_local_failed = (not moondream_ok) and (not qwen_ok)
        if both_local_failed:
            fallback_prompt = (
                f"{user_message.strip() or '请根据上下文回答。'}\n"
                f"（用户附带了 {len(image_list)} 张图片；本地 moondream2 与 qwen2.5-7b 均不可用。）"
            )
            pipeline_name = "fallback:gpt-sol(both-local-failed)"
        else:
            fallback_prompt = (
                "下面是图片解析结果与用户原始问题，请综合回答用户问题。\n\n"
                f"【图片解析】\n{image_desc or '（无）'}\n\n"
                f"【用户问题】\n{user_message.strip() or '（无文字）'}"
            )
            pipeline_name = "fallback:gpt-sol(after-qwen-failed)"

        try:
            sol = _chat_text(
                model_key="gpt-sol",
                user_message=fallback_prompt,
                system_message=system,
                temperature=temperature,
                settings=cfg,
            )
            steps.append(
                {
                    "step": "fallback",
                    "ok": True,
                    "key": sol["key"],
                    "model": sol["model"],
                    "both_local_failed": both_local_failed,
                }
            )
            return {
                "key": sol["key"],
                "provider": sol["provider"],
                "model": sol["model"],
                "reply": sol["reply"],
                "pipeline": pipeline_name,
                "image_description": image_desc,
                "fallback": True,
                "steps": steps,
                "errors": errors,
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"gpt-sol: {exc}")
            steps.append({"step": "fallback", "ok": False, "error": str(exc)})
            raise RuntimeError("本地模型与 gpt-sol 均调用失败: " + " | ".join(errors)) from exc
    # ---------- 无图：qwen → gpt-sol ----------
    try:
        qwen = _chat_text(
            model_key="qwen2.5-7b",
            user_message=user_message,
            system_message=system,
            temperature=temperature,
            settings=cfg,
        )
        steps.append({"step": "answer", "ok": True, "key": qwen["key"], "model": qwen["model"]})
        return {
            **qwen,
            "pipeline": "qwen2.5-7b",
            "image_description": "",
            "fallback": False,
            "steps": steps,
            "errors": errors,
        }
    except Exception as exc:  # noqa: BLE001
        errors.append(f"qwen2.5-7b: {exc}")
        steps.append({"step": "answer", "ok": False, "error": str(exc)})

    sol = _chat_text(
        model_key="gpt-sol",
        user_message=user_message,
        system_message=system,
        temperature=temperature,
        settings=cfg,
    )
    steps.append({"step": "fallback", "ok": True, "key": sol["key"], "model": sol["model"]})
    return {
        **sol,
        "pipeline": "fallback:gpt-sol",
        "image_description": "",
        "fallback": True,
        "steps": steps,
        "errors": errors,
    }
