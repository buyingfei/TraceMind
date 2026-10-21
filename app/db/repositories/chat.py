"""对话会话 / 消息 CRUD（本系统 MySQL）。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import ChatConversation, ChatMessage, ChatMessageAsset
from app.infra.db import get_session_factory


def asset_public_url(relative_path: str) -> str:
    safe = relative_path.replace("\\", "/").lstrip("/")
    return f"/api/v1/chats/files/{safe}"


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


class ChatRepository:
    """封装 chat_conversation / chat_message / chat_message_asset 读写。"""

    def __init__(self, session: Session | None = None):
        self._external = session
        self._owned: Session | None = None

    def _session(self) -> Session:
        if self._external is not None:
            return self._external
        if self._owned is None:
            self._owned = get_session_factory()()
        return self._owned

    def close(self) -> None:
        if self._owned is not None:
            self._owned.close()
            self._owned = None

    def __enter__(self) -> ChatRepository:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def list_conversations(self, *, limit: int = 100) -> list[dict[str, Any]]:
        session = self._session()
        rows = session.scalars(
            select(ChatConversation)
            .where(ChatConversation.status != "deleted")
            .order_by(ChatConversation.last_message_at.desc(), ChatConversation.id.desc())
            .limit(limit)
        ).all()
        return [_conv_to_dict(r) for r in rows]

    def create_conversation(self, *, title: str = "新对话") -> dict[str, Any]:
        session = self._session()
        row = ChatConversation(title=(title or "新对话")[:256], status="active", message_count=0)
        session.add(row)
        session.commit()
        session.refresh(row)
        return _conv_to_dict(row)

    def get_conversation(self, conversation_id: int) -> dict[str, Any] | None:
        session = self._session()
        row = session.get(ChatConversation, conversation_id)
        if row is None or row.status == "deleted":
            return None
        return _conv_to_dict(row)

    def soft_delete_conversation(self, conversation_id: int) -> bool:
        session = self._session()
        row = session.get(ChatConversation, conversation_id)
        if row is None:
            return False
        row.status = "deleted"
        session.commit()
        return True

    def list_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        session = self._session()
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

    def add_user_message(
        self,
        conversation_id: int,
        *,
        content: str,
        client_msg_id: str,
        relative_paths: list[str] | None = None,
        resolve_upload: Callable[[str], Any] | None = None,
        guess_mime: Callable[[Any], str] | None = None,
    ) -> dict[str, Any]:
        """持久化用户消息；附件元数据依赖 resolve_upload(rel) → Path。"""
        text = (content or "").strip()
        paths = [p.replace("\\", "/").lstrip("/") for p in (relative_paths or []) if p]
        if not text and not paths:
            raise ValueError("content 与图片不能同时为空")
        if not (client_msg_id or "").strip():
            client_msg_id = f"cli_{uuid.uuid4().hex}"

        session = self._session()
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
                if resolve_upload is None:
                    raise ValueError("缺少 resolve_upload，无法绑定附件")
                full = resolve_upload(rel)
                mime = "image/png"
                if guess_mime is not None:
                    mime = guess_mime(full)
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

    def add_assistant_message(
        self,
        conversation_id: int,
        *,
        content: str,
        status: str = "normal",
        error_message: str = "",
    ) -> dict[str, Any]:
        session = self._session()
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


def list_conversations(*, limit: int = 100) -> list[dict[str, Any]]:
    with ChatRepository() as repo:
        return repo.list_conversations(limit=limit)


def create_conversation(*, title: str = "新对话") -> dict[str, Any]:
    with ChatRepository() as repo:
        return repo.create_conversation(title=title)


def get_conversation(conversation_id: int) -> dict[str, Any] | None:
    with ChatRepository() as repo:
        return repo.get_conversation(conversation_id)


def soft_delete_conversation(conversation_id: int) -> bool:
    with ChatRepository() as repo:
        return repo.soft_delete_conversation(conversation_id)


def list_messages(conversation_id: int) -> list[dict[str, Any]]:
    with ChatRepository() as repo:
        return repo.list_messages(conversation_id)


def add_user_message(
    conversation_id: int,
    *,
    content: str,
    client_msg_id: str,
    relative_paths: list[str] | None = None,
    resolve_upload: Callable[[str], Any] | None = None,
    guess_mime: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    with ChatRepository() as repo:
        return repo.add_user_message(
            conversation_id,
            content=content,
            client_msg_id=client_msg_id,
            relative_paths=relative_paths,
            resolve_upload=resolve_upload,
            guess_mime=guess_mime,
        )


def add_assistant_message(
    conversation_id: int,
    *,
    content: str,
    status: str = "normal",
    error_message: str = "",
) -> dict[str, Any]:
    with ChatRepository() as repo:
        return repo.add_assistant_message(
            conversation_id,
            content=content,
            status=status,
            error_message=error_message,
        )
