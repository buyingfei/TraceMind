"""OpenAI 兼容聊天服务（对应 Java Spring AI OpenAI 配置）。"""

from functools import lru_cache

from openai import OpenAI

from app.core.config import Settings, get_settings


@lru_cache
def get_openai_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY，请写入 .env")
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.normalized_openai_base_url(),
    )


def chat(
    user_message: str,
    *,
    system_message: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> str:
    """调用 gpt 兼容接口，返回助手文本。"""
    cfg = settings or get_settings()
    client = get_openai_client()

    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model or cfg.openai_model,
        temperature=cfg.openai_temperature if temperature is None else temperature,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content or ""
