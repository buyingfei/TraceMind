"""多模型联调：gpt-sol / moondream2 / qwen2.5-7b。"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services import llm_service

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    system: str | None = Field(default=None, description="可选系统提示词")
    model: str | None = Field(
        default=None,
        description="模型 key：gpt-sol | moondream2 | qwen2.5-7b；不传用默认",
    )


class ChatResponse(BaseModel):
    key: str
    provider: str
    model: str
    reply: str


@router.get("/llm/models")
def llm_models() -> dict:
    """列出可用聊天模型（共 3 个）。"""
    return {"models": llm_service.list_chat_models()}


@router.get("/llm/config")
def llm_config() -> dict:
    """查看默认模型与网关/Ollama 配置（不返回完整 key）。"""
    settings = get_settings()
    key = settings.openai_api_key
    masked = f"{key[:7]}...{key[-4:]}" if len(key) > 12 else "(empty)"
    return {
        "default_chat_model": settings.default_chat_model,
        "openai": {
            "base_url": settings.normalized_openai_base_url(),
            "model": settings.openai_model,
            "temperature": settings.openai_temperature,
            "api_key": masked,
        },
        "ollama": {
            "base_url": settings.ollama_base_url,
            "moondream2": settings.ollama_chat_model_moondream,
            "qwen2.5-7b": settings.ollama_chat_model_qwen,
        },
        "models": llm_service.list_chat_models(settings=settings),
    }


@router.post("/llm/chat", response_model=ChatResponse)
def llm_chat(body: ChatRequest) -> ChatResponse:
    try:
        result = llm_service.chat(
            body.message,
            system_message=body.system or "你是 diagnose-agent 助手，用简洁中文回答。",
            model_key=body.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败: {exc}") from exc
    return ChatResponse(**result)
