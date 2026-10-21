"""对话编排：本地图片上传 + Repository CRUD + LLM/Agent。"""

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

from app.agents.diagnose import chat_runtime
from app.core.config import get_settings
from app.db.repositories import chat as chat_repo
from app.db.repositories.chat import asset_public_url
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


def list_conversations(*, limit: int = 100) -> list[dict[str, Any]]:
    return chat_repo.list_conversations(limit=limit)


def create_conversation(*, title: str = "新对话") -> dict[str, Any]:
    return chat_repo.create_conversation(title=title)


def get_conversation(conversation_id: int) -> dict[str, Any] | None:
    return chat_repo.get_conversation(conversation_id)


def soft_delete_conversation(conversation_id: int) -> bool:
    return chat_repo.soft_delete_conversation(conversation_id)


def list_messages(conversation_id: int) -> list[dict[str, Any]]:
    return chat_repo.list_messages(conversation_id)


def add_user_message(
    conversation_id: int,
    *,
    content: str,
    client_msg_id: str,
    relative_paths: list[str] | None = None,
) -> dict[str, Any]:
    def _guess_mime(full: Path) -> str:
        return mimetypes.guess_type(str(full))[0] or "image/png"

    return chat_repo.add_user_message(
        conversation_id,
        content=content,
        client_msg_id=client_msg_id,
        relative_paths=relative_paths,
        resolve_upload=_safe_resolve_under_uploads,
        guess_mime=_guess_mime,
    )


def add_assistant_message(
    conversation_id: int,
    *,
    content: str,
    status: str = "normal",
    error_message: str = "",
) -> dict[str, Any]:
    return chat_repo.add_assistant_message(
        conversation_id,
        content=content,
        status=status,
        error_message=error_message,
    )


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
            llm_result = chat_runtime.run_chat_turn(
                conversation_id,
                user_text=text,
                images=image_data_urls,
            )
            llm_meta = {
                "key": llm_result.get("key"),
                "model": llm_result.get("model"),
                "pipeline": llm_result.get("pipeline"),
                "fallback": llm_result.get("fallback"),
                "image_description": llm_result.get("image_description"),
                "errors": llm_result.get("errors"),
                "memory": True,
            }
            reply = llm_result.get("reply") or ""
            status = "normal"
            err = ""
        except Exception as exc:  # noqa: BLE001
            logger.exception("对话 Agent 失败 conversation_id=%s", conversation_id)
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
        for ev in chat_runtime.iter_chat_turn_stream(
            conversation_id,
            user_text=text,
            images=image_data_urls,
        ):
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
            "memory": True,
            "code_hits": (done_payload or {}).get("code_hits"),
            "hit_count": (done_payload or {}).get("hit_count"),
            "assistant_message": persisted["assistant_message"],
            "conversation": persisted["conversation"],
            "user_message": saved["user_message"],
        },
    )
