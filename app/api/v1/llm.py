"""模型联调接口（验证网关 + gpt-5.6-sol）。"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services import llm_service

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    system: str | None = Field(default=None, description="可选系统提示词")


class ChatResponse(BaseModel):
    model: str
    reply: str


@router.get("/llm/config")
def llm_config() -> dict[str, str | float]:
    """查看当前模型配置（不返回完整 key）。"""
    settings = get_settings()
    key = settings.openai_api_key
    masked = f"{key[:7]}...{key[-4:]}" if len(key) > 12 else "(empty)"
    return {
        "base_url": settings.normalized_openai_base_url(),
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "api_key": masked,
    }


@router.post("/llm/chat", response_model=ChatResponse)
def llm_chat(body: ChatRequest) -> ChatResponse:
    settings = get_settings()
    try:
        reply = llm_service.chat(
            body.message,
            system_message=body.system or "你是 diagnose-agent 助手，用简洁中文回答。",
        )
    except Exception as exc:  # noqa: BLE001 - 联调阶段直接返回错误信息
        raise HTTPException(status_code=502, detail=f"模型调用失败: {exc}") from exc
    return ChatResponse(model=settings.openai_model, reply=reply)
