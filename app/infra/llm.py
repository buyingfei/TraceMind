"""OpenAI 兼容 / Ollama 客户端单例。"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.core.config import get_settings


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


def client_for_provider(provider: str) -> OpenAI:
    if provider == "openai":
        return get_openai_client()
    if provider == "ollama":
        return get_ollama_openai_client()
    raise ValueError(f"不支持的 provider: {provider}")
