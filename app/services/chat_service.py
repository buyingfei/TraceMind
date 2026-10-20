"""对话持久化：MySQL + 本地图片 + LLM 编排。"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import ChatConversation, ChatMessage, ChatMessageAsset
from app.db.session import get_session_factory
from app.services import llm_service

logger = logging.getLogger(__name__)

_ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
_MAX_BYTES = 5 * 1024 * 1024


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def chat_uploads_root() -> Path:
    settings = get_settings()
    root = Path(settings.chat_upload_dir)
    if not root.is_absolute():
        root = _project_root() / root
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def asset_public_url(relative_path: str) -> str:
    # 由前端/网关走 /api/v1 前缀
    safe = relative_path.replace("\\", "/").lstrip("/")
    return f"/api/v1/chats/files/{safe}"


def _safe_resolve_under_uploads(relative_path: str) -> Path:
    root = chat_uploads_root()
    rel = Path(relative_path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise FileNotFoundError("非法路径")
    full = (root / rel).resolve()
    if not str(full).startswith(str(root)):
        raise FileNotFoundError("非法路径")
    if not full.is_file():
        raise FileNotFoundError(f"文件不存在: {relative_path}")
    return full


def resolve_asset_file(relative_path: str) -> Path:
    return _safe_resolve_under_uploads(relative_path)


def _ext_for_mime(mime: str) -> str:
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    return mapping.get(mime.lower(), mimetypes.guess_extension(mime) or ".bin")


def save_upload_bytes(
    data: bytes,
    *,
    mime_type: str,
    conversation_id: int | None = None,
) -> dict[str, Any]:
    mime = (mime_type or "application/octet-stream").split(";")[0].strip().lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in _ALLOWED_MIME:
        raise ValueError(f"不支持的图片类型: {mime}")
    if len(data) > _MAX_BYTES:
        raise ValueError(f"图片超过 {_MAX_BYTES} 字节上限")

    day = datetime.now().strftime("%Y%m%d")
    folder = str(conversation_id or "pending")
    name = f"{uuid.uuid4().hex}{_ext_for_mime(mime)}"
    relative = f"{folder}/{day}/{name}".replace("\\", "/")
    dest = chat_uploads_root() / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {
        "relative_path": relative,
        "url": asset_public_url(relative),
        "mime_type": mime,
        "size_bytes": len(data),
    }


def save_upload_data_url(data_url: str, *, conversation_id: int | None = None) -> dict[str, Any]:
    mime, raw = llm_service.normalize_image_input(data_url)
    data = base64.b64decode(raw)
    return save_upload_bytes(data, mime_type=mime, conversation_id=conversation_id)


def file_to_data_url(relative_path: str) -> str:
    path = _safe_resolve_under_uploads(relative_path)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _conv_to_dict(row: ChatConversation) -> dict[str, Any]:
    return {
        "id": row.id,
        "title": row.title,
        "status": row.status,
        "message_count": row.message_count,
        "last_message_at": row.last_message_at.isoformat(sep=" ", timespec="seconds")
        if row.last_message_at
        else None,
        "updated_at": row.updated_at.isoformat(sep=" ", timespec="seconds") if row.updated_at else None,
        "created_at": row.created_at.isoformat(sep=" ", timespec="seconds") if row.created_at else None,
        "context_repo": row.context_repo,
        "context_image_tag": row.context_image_tag,
        "context_index_version": row.context_index_version,
    }


def list_conversations(*, limit: int = 100) -> list[dict[str, Any]]:
    factory = get_session_factory()
    session = factory()
    try:
        rows = session.scalars(
            select(ChatConversation)
            .where(ChatConversation.status != "deleted")
            .order_by(ChatConversation.last_message_at.desc(), ChatConversation.id.desc())
            .limit(limit)
        ).all()
        return [_conv_to_dict(r) for r in rows]
    finally:
        session.close()


def create_conversation(*, title: str = "新对话") -> dict[str, Any]:
    factory = get_session_factory()
    session = factory()
    try:
        row = ChatConversation(title=(title or "新对话")[:256], status="active", message_count=0)
        session.add(row)
        session.commit()
        session.refresh(row)
        return _conv_to_dict(row)
    finally:
        session.close()


def get_conversation(conversation_id: int) -> dict[str, Any] | None:
    factory = get_session_factory()
    session = factory()
    try:
        row = session.get(ChatConversation, conversation_id)
        if row is None or row.status == "deleted":
            return None
        return _conv_to_dict(row)
    finally:
        session.close()


def soft_delete_conversation(conversation_id: int) -> bool:
    factory = get_session_factory()
    session = factory()
    try:
        row = session.get(ChatConversation, conversation_id)
        if row is None:
            return False
        row.status = "deleted"
        session.commit()
        return True
    finally:
        session.close()


def _message_to_dict(msg: ChatMessage, assets: list[ChatMessageAsset]) -> dict[str, Any]:
    return {
        "id": msg.id,
        "conversation_id": msg.conversation_id,
        "role": msg.role,
        "content": msg.content,
        "content_type": msg.content_type,
        "seq": msg.seq,
        "client_msg_id": msg.client_msg_id,
        "status": msg.status,
        "error_message": msg.error_message or "",
        "created_at": msg.created_at.isoformat(sep=" ", timespec="seconds") if msg.created_at else None,
        "images": [
            {
                "id": a.id,
                "relative_path": a.relative_path,
                "url": asset_public_url(a.relative_path),
                "mime_type": a.mime_type,
                "sort_no": a.sort_no,
            }
            for a in sorted(assets, key=lambda x: x.sort_no)
        ],
    }


def list_messages(conversation_id: int) -> list[dict[str, Any]]:
    factory = get_session_factory()
    session = factory()
    try:
        conv = session.get(ChatConversation, conversation_id)
        if conv is None or conv.status == "deleted":
            raise FileNotFoundError(f"会话不存在: {conversation_id}")
        msgs = session.scalars(
            select(ChatMessage)
            .where(
                ChatMessage.conversation_id == conversation_id,
                ChatMessage.status != "deleted",
            )
            .order_by(ChatMessage.seq.asc(), ChatMessage.id.asc())
        ).all()
        if not msgs:
            return []
        ids = [m.id for m in msgs]
        assets = session.scalars(select(ChatMessageAsset).where(ChatMessageAsset.message_id.in_(ids))).all()
        by_msg: dict[int, list[ChatMessageAsset]] = {}
        for a in assets:
            by_msg.setdefault(a.message_id, []).append(a)
        return [_message_to_dict(m, by_msg.get(m.id, [])) for m in msgs]
    finally:
        session.close()


def add_user_message(
    conversation_id: int,
    *,
    content: str,
    client_msg_id: str,
    relative_paths: list[str] | None = None,
) -> dict[str, Any]:
    """仅持久化用户消息；返回 user_message / conversation / paths / deduplicated。"""
    text = (content or "").strip()
    paths = [p.replace("\\", "/").lstrip("/") for p in (relative_paths or []) if p]
    if not text and not paths:
        raise ValueError("content 与图片不能同时为空")
    if not (client_msg_id or "").strip():
        client_msg_id = f"cli_{uuid.uuid4().hex}"

    factory = get_session_factory()
    session = factory()
    try:
        conv = session.get(ChatConversation, conversation_id)
        if conv is None or conv.status == "deleted":
            raise FileNotFoundError(f"会话不存在: {conversation_id}")

        existing = session.scalar(
            select(ChatMessage).where(
                ChatMessage.conversation_id == conversation_id,
                ChatMessage.client_msg_id == client_msg_id,
            )
        )
        if existing is not None:
            assets = session.scalars(
                select(ChatMessageAsset).where(ChatMessageAsset.message_id == existing.id)
            ).all()
            return {
                "user_message": _message_to_dict(existing, list(assets)),
                "conversation": _conv_to_dict(conv),
                "paths": [a.relative_path for a in assets],
                "text": existing.content or "",
                "deduplicated": True,
            }

        max_seq = session.scalar(
            select(func.max(ChatMessage.seq)).where(ChatMessage.conversation_id == conversation_id)
        )
        next_seq = int(max_seq or 0) + 1

        user_msg = ChatMessage(
            conversation_id=conversation_id,
            role="user",
            content=text,
            content_type="mixed" if paths else "text",
            seq=next_seq,
            client_msg_id=client_msg_id,
            status="normal",
        )
        session.add(user_msg)
        session.flush()

        asset_rows: list[ChatMessageAsset] = []
        for i, rel in enumerate(paths):
            full = _safe_resolve_under_uploads(rel)
            mime = mimetypes.guess_type(str(full))[0] or "image/png"
            asset = ChatMessageAsset(
                message_id=user_msg.id,
                asset_type="image",
                relative_path=rel,
                mime_type=mime,
                size_bytes=full.stat().st_size,
                sort_no=i,
            )
            session.add(asset)
            asset_rows.append(asset)

        if conv.title == "新对话":
            conv.title = (text or "图片消息").replace("\n", " ")[:18] or "图片消息"
        conv.message_count = int(conv.message_count or 0) + 1
        conv.last_message_at = datetime.now()
        session.commit()
        session.refresh(user_msg)
        for a in asset_rows:
            session.refresh(a)
        return {
            "user_message": _message_to_dict(user_msg, asset_rows),
            "conversation": _conv_to_dict(conv),
            "paths": paths,
            "text": text,
            "deduplicated": False,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def add_assistant_message(
    conversation_id: int,
    *,
    content: str,
    status: str = "normal",
    error_message: str = "",
) -> dict[str, Any]:
    factory = get_session_factory()
    session = factory()
    try:
        conv = session.get(ChatConversation, conversation_id)
        if conv is None or conv.status == "deleted":
            raise FileNotFoundError(f"会话不存在: {conversation_id}")
        max_seq = session.scalar(
            select(func.max(ChatMessage.seq)).where(ChatMessage.conversation_id == conversation_id)
        )
        assistant = ChatMessage(
            conversation_id=conversation_id,
            role="assistant",
            content=content or "",
            content_type="text",
            seq=int(max_seq or 0) + 1,
            client_msg_id=f"srv_{uuid.uuid4().hex}",
            status=status,
            error_message=(error_message or "")[:500],
        )
        session.add(assistant)
        conv.message_count = int(conv.message_count or 0) + 1
        conv.last_message_at = datetime.now()
        session.commit()
        session.refresh(assistant)
        session.refresh(conv)
        return {
            "assistant_message": _message_to_dict(assistant, []),
            "conversation": _conv_to_dict(conv),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def send_message(
    conversation_id: int,
    *,
    content: str,
    client_msg_id: str,
    relative_paths: list[str] | None = None,
    call_llm: bool = True,
) -> dict[str, Any]:
    """写入用户消息（及附件）→ 调 LLM → 写入助手消息。"""
    saved = add_user_message(
        conversation_id,
        content=content,
        client_msg_id=client_msg_id,
        relative_paths=relative_paths,
    )
    if saved.get("deduplicated"):
        return {
            "user_message": saved["user_message"],
            "assistant_message": None,
            "conversation": saved["conversation"],
            "llm": None,
            "deduplicated": True,
        }

    user_dict = saved["user_message"]
    conv_dict = saved["conversation"]
    paths = saved["paths"]
    text = saved["text"]

    assistant_dict = None
    llm_meta = None
    if call_llm:
        try:
            image_data_urls = [file_to_data_url(p) for p in paths]
            llm_result = llm_service.chat_with_pipeline(
                text,
                images=image_data_urls,
            )
            llm_meta = {
                "key": llm_result.get("key"),
                "model": llm_result.get("model"),
                "pipeline": llm_result.get("pipeline"),
                "fallback": llm_result.get("fallback"),
                "image_description": llm_result.get("image_description"),
                "errors": llm_result.get("errors"),
            }
            reply = llm_result.get("reply") or ""
            status = "normal"
            err = ""
        except Exception as exc:  # noqa: BLE001
            logger.exception("对话 LLM 失败 conversation_id=%s", conversation_id)
            reply = f"模型调用失败：{exc}"
            status = "failed"
            err = str(exc)[:500]
            llm_meta = {"error": str(exc)}

        persisted = add_assistant_message(
            conversation_id,
            content=reply,
            status=status,
            error_message=err,
        )
        assistant_dict = persisted["assistant_message"]
        conv_dict = persisted["conversation"]

    return {
        "user_message": user_dict,
        "assistant_message": assistant_dict,
        "conversation": conv_dict,
        "llm": llm_meta,
        "deduplicated": False,
    }


def _sse_pack(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def iter_message_sse(
    conversation_id: int,
    *,
    content: str,
    client_msg_id: str,
    relative_paths: list[str] | None = None,
) -> Iterator[str]:
    """SSE：先落用户消息，再流式生成助手回复并落库。"""
    try:
        saved = add_user_message(
            conversation_id,
            content=content,
            client_msg_id=client_msg_id,
            relative_paths=relative_paths,
        )
    except Exception as exc:  # noqa: BLE001
        yield _sse_pack("error", {"message": str(exc)})
        return

    yield _sse_pack(
        "meta",
        {
            "stage": "user_saved",
            "user_message": saved["user_message"],
            "conversation": saved["conversation"],
            "deduplicated": saved.get("deduplicated", False),
        },
    )

    if saved.get("deduplicated"):
        yield _sse_pack(
            "done",
            {
                "deduplicated": True,
                "user_message": saved["user_message"],
                "conversation": saved["conversation"],
            },
        )
        return

    paths = saved["paths"]
    text = saved["text"]
    try:
        image_data_urls = [file_to_data_url(p) for p in paths]
    except Exception as exc:  # noqa: BLE001
        yield _sse_pack("error", {"message": f"读取图片失败: {exc}"})
        return

    full_reply = ""
    done_payload: dict[str, Any] | None = None
    try:
        for ev in llm_service.iter_pipeline_stream(text, images=image_data_urls):
            et = ev.get("type")
            if et == "meta":
                yield _sse_pack("meta", {k: v for k, v in ev.items() if k != "type"})
            elif et == "delta":
                yield _sse_pack("delta", {"text": ev.get("text") or ""})
            elif et == "error":
                msg = ev.get("message") or "生成失败"
                persisted = add_assistant_message(
                    conversation_id,
                    content=f"模型调用失败：{msg}",
                    status="failed",
                    error_message=str(msg)[:500],
                )
                yield _sse_pack(
                    "error",
                    {
                        "message": msg,
                        "assistant_message": persisted["assistant_message"],
                        "conversation": persisted["conversation"],
                    },
                )
                return
            elif et == "done":
                full_reply = ev.get("reply") or ""
                done_payload = ev
    except Exception as exc:  # noqa: BLE001
        logger.exception("SSE 流式失败 conversation_id=%s", conversation_id)
        persisted = add_assistant_message(
            conversation_id,
            content=f"模型调用失败：{exc}",
            status="failed",
            error_message=str(exc)[:500],
        )
        yield _sse_pack(
            "error",
            {
                "message": str(exc),
                "assistant_message": persisted["assistant_message"],
                "conversation": persisted["conversation"],
            },
        )
        return

    persisted = add_assistant_message(
        conversation_id,
        content=full_reply,
        status="normal",
    )
    yield _sse_pack(
        "done",
        {
            "key": (done_payload or {}).get("key"),
            "model": (done_payload or {}).get("model"),
            "provider": (done_payload or {}).get("provider"),
            "pipeline": (done_payload or {}).get("pipeline"),
            "fallback": (done_payload or {}).get("fallback"),
            "image_description": (done_payload or {}).get("image_description"),
            "errors": (done_payload or {}).get("errors"),
            "assistant_message": persisted["assistant_message"],
            "conversation": persisted["conversation"],
            "user_message": saved["user_message"],
        },
    )
