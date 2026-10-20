"""多模型聊天：OpenAI 网关 gpt-sol + 本地 Ollama（moondream2 / qwen2.5-7b）。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.core.config import Settings, get_settings

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


def chat(
    user_message: str,
    *,
    system_message: str | None = None,
    model_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """调用指定聊天模型，返回 {key, provider, model, reply}。

    - model_key: gpt-sol | moondream2 | qwen2.5-7b
    - model: 兼容旧参数，若传入且像网关模型名则走 openai；优先使用 model_key
    """
    cfg = settings or get_settings()
    key = (model_key or cfg.default_chat_model or "gpt-sol").strip()
    if key not in CHAT_MODEL_CATALOG:
        # 兼容旧调用：直接传网关模型名
        if model and model == cfg.openai_model:
            key = "gpt-sol"
        else:
            raise ValueError(f"未知模型 key: {key}，可选: {', '.join(CHAT_MODEL_CATALOG)}")

    meta = CHAT_MODEL_CATALOG[key]
    provider = meta["provider"]
    model_name = model or _resolve_model_name(key, cfg)
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
        "key": key,
        "provider": provider,
        "model": model_name,
        "reply": content or "",
    }
