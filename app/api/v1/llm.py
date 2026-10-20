"""多模型联调：带图 PaddleOCR→qwen2.5-7b，失败 gpt-sol 兜底。"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services import llm_service

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(default="", description="用户文字问题")
    system: str | None = Field(default=None, description="可选系统提示词")
    model: str | None = Field(
        default=None,
        description="仅 use_pipeline=false 且无图时生效：gpt-sol|qwen2.5-7b",
    )
    images: list[str] = Field(
        default_factory=list,
        description="图片 data URL 或纯 base64；有图则走 PaddleOCR→qwen2.5-7b",
    )
    use_pipeline: bool = Field(
        default=True,
        description="默认编排；false 且无图时按 model 单模型调用",
    )


class ChatResponse(BaseModel):
    key: str
    provider: str
    model: str
    reply: str
    pipeline: str | None = None
    image_description: str | None = None
    fallback: bool = False
    steps: list[dict] | None = None
    errors: list[str] | None = None


@router.get("/llm/models")
def llm_models() -> dict:
    """列出可用聊天模型。"""
    return {"models": llm_service.list_chat_models()}


@router.get("/llm/config")
def llm_config() -> dict:
    """查看默认模型与网关/Ollama 配置（不返回完整 key）。"""
    settings = get_settings()
    key = settings.openai_api_key
    masked = f"{key[:7]}...{key[-4:]}" if len(key) > 12 else "(empty)"
    return {
        "default_chat_model": settings.default_chat_model,
        "pipeline": {
            "with_images": "paddleocr -> qwen2.5-7b；失败则 gpt-sol",
            "without_images": "qwen2.5-7b -> 失败则 gpt-sol",
        },
        "openai": {
            "base_url": settings.normalized_openai_base_url(),
            "model": settings.openai_model,
            "temperature": settings.openai_temperature,
            "api_key": masked,
        },
        "ollama": {
            "base_url": settings.ollama_base_url,
            "qwen2.5-7b": settings.ollama_chat_model_qwen,
        },
        "ocr": "paddleocr-ch",
        "models": llm_service.list_chat_models(settings=settings),
    }


@router.post("/llm/chat", response_model=ChatResponse)
def llm_chat(body: ChatRequest) -> ChatResponse:
    if not (body.message or "").strip() and not body.images:
        raise HTTPException(status_code=400, detail="message 与 images 不能同时为空")

    system = body.system or "你是 diagnose-agent 助手，用简洁中文回答。"
    try:
        if body.images or body.use_pipeline:
            result = llm_service.chat_with_pipeline(
                body.message,
                images=body.images,
                system_message=system,
            )
        else:
            result = llm_service.chat(
                body.message,
                system_message=system,
                model_key=body.model,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败: {exc}") from exc

    return ChatResponse(
        key=result.get("key", ""),
        provider=result.get("provider", ""),
        model=result.get("model", ""),
        reply=result.get("reply", ""),
        pipeline=result.get("pipeline"),
        image_description=result.get("image_description") or None,
        fallback=bool(result.get("fallback")),
        steps=result.get("steps"),
        errors=result.get("errors") or None,
    )
