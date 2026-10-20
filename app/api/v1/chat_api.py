"""对话 API：会话 / 消息 / 本地图片。"""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.services import chat_service

router = APIRouter(prefix="/chats", tags=["chats"])


class CreateConversationRequest(BaseModel):
    title: str = Field(default="新对话")


class SendMessageRequest(BaseModel):
    content: str = Field(default="")
    client_msg_id: str = Field(..., description="前端幂等键")
    relative_paths: list[str] = Field(default_factory=list, description="已上传图片相对路径")
    call_llm: bool = Field(default=True)


class UploadDataUrlRequest(BaseModel):
    data_url: str
    conversation_id: int | None = None


@router.get("")
def list_chats() -> dict:
    return {"items": chat_service.list_conversations()}


@router.post("")
def create_chat(body: CreateConversationRequest | None = None) -> dict:
    body = body or CreateConversationRequest()
    return chat_service.create_conversation(title=body.title)


# 固定路径须写在 /{conversation_id} 之前，避免被当成 id
@router.post("/assets/upload")
async def upload_asset(
    file: UploadFile = File(...),
    conversation_id: int | None = Form(default=None),
) -> dict:
    data = await file.read()
    mime = file.content_type or "application/octet-stream"
    try:
        return chat_service.save_upload_bytes(
            data,
            mime_type=mime,
            conversation_id=conversation_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/assets/upload-data-url")
def upload_data_url(body: UploadDataUrlRequest) -> dict:
    try:
        return chat_service.save_upload_data_url(
            body.data_url,
            conversation_id=body.conversation_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/files/{file_path:path}")
def get_file(file_path: str):
    try:
        full = chat_service.resolve_asset_file(file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    mime = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
    return FileResponse(full, media_type=mime)


@router.get("/{conversation_id}")
def get_chat(conversation_id: int) -> dict:
    row = chat_service.get_conversation(conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return row


@router.delete("/{conversation_id}")
def delete_chat(conversation_id: int) -> dict:
    ok = chat_service.soft_delete_conversation(conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@router.get("/{conversation_id}/messages")
def get_messages(conversation_id: int) -> dict:
    try:
        items = chat_service.list_messages(conversation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"items": items}


@router.post("/{conversation_id}/messages")
def post_message(conversation_id: int, body: SendMessageRequest) -> dict:
    try:
        return chat_service.send_message(
            conversation_id,
            content=body.content,
            client_msg_id=body.client_msg_id,
            relative_paths=body.relative_paths,
            call_llm=body.call_llm,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"发送失败: {exc}") from exc


@router.post("/{conversation_id}/messages/stream")
def post_message_stream(conversation_id: int, body: SendMessageRequest):
    """SSE 流式回答（打字机）。事件：meta / delta / done / error。"""
    if not (body.content or "").strip() and not body.relative_paths:
        raise HTTPException(status_code=400, detail="content 与图片不能同时为空")

    generator = chat_service.iter_message_sse(
        conversation_id,
        content=body.content,
        client_msg_id=body.client_msg_id,
        relative_paths=body.relative_paths,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
