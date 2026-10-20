"""将 Java 源码切块、向量化并写入 Milvus / MySQL meta。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from datetime import datetime

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import CodeChunkMeta, CodeIndexVersion, CodeRepo
from app.db.session import get_session_factory
from app.services import milvus_service
from app.services.treesitter_java_splitter import (
    load_code_index_config,
    resolve_preview_file,
    split_java_file,
)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _build_milvus_id(
    *,
    repo: str,
    image_tag: str,
    index_version: str,
    path: str,
    start_line: int,
    end_line: int,
    node_type: str,
) -> str:
    raw = f"{repo}|{image_tag}|{index_version}|{path}|{start_line}|{end_line}|{node_type}"
    return _md5(raw)


def _module_of(path: str) -> str:
    parts = Path(path).parts
    return parts[0] if parts else ""


def load_current_index_context(repo_code: str = "wintogether-bff") -> dict[str, Any]:
    """从 MySQL 读取当前 is_current 的索引版本；没有则抛错。"""
    factory = get_session_factory()
    session = factory()
    try:
        repo = session.scalar(select(CodeRepo).where(CodeRepo.repo_code == repo_code))
        if repo is None:
            raise RuntimeError(f"MySQL 中不存在 code_repo: {repo_code}，请先手工写入仓库与版本数据")
        version = session.scalar(
            select(CodeIndexVersion).where(
                CodeIndexVersion.repo_id == repo.id,
                CodeIndexVersion.is_current == 1,
            )
        )
        if version is None:
            raise RuntimeError(f"MySQL 中不存在 is_current=1 的 code_index_version（repo={repo_code}）")
        return {
            "repo_id": repo.id,
            "repo": repo.repo_code,
            "local_root": repo.local_root,
            "branch": version.branch,
            "image_tag": version.image_tag,
            "git_commit": version.git_commit,
            "index_version": version.index_version,
            "index_version_id": version.id,
            "milvus_collection": version.milvus_collection,
        }
    finally:
        session.close()


def index_java_file(
    relative_path: str | None = None,
    *,
    recreate_collection: bool = False,
) -> dict[str, Any]:
    """切块指定 Java 文件 → Ollama 向量 → Milvus + code_chunk_meta。"""
    settings = get_settings()
    cfg = load_code_index_config()
    ctx = load_current_index_context(cfg.get("repo", {}).get("name", "wintogether-bff"))

    if relative_path:
        file_path = Path(ctx["local_root"]) / relative_path
        rel = relative_path.replace("/", "\\")
    else:
        file_path = resolve_preview_file(cfg)
        rel = str(cfg.get("preview", {}).get("file") or file_path.name)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    splitter_cfg = cfg.get("splitter") or {}
    chunks = split_java_file(
        file_path,
        node_types=splitter_cfg.get("node_types"),
        include_preamble=bool(splitter_cfg.get("include_preamble", True)),
        max_chars=int(splitter_cfg.get("max_chars", 4000)),
    )

    if recreate_collection:
        milvus_service.ensure_collection(settings=settings, drop_if_exists=True)
    else:
        milvus_service.ensure_collection(settings=settings, drop_if_exists=False)

    milvus_rows: list[dict[str, Any]] = []
    meta_rows: list[CodeChunkMeta] = []
    rel_norm = rel.replace("\\", "/")

    for chunk in chunks:
        content_hash = _md5(chunk.text)
        milvus_id = _build_milvus_id(
            repo=ctx["repo"],
            image_tag=ctx["image_tag"],
            index_version=ctx["index_version"],
            path=rel_norm,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            node_type=chunk.node_type,
        )
        milvus_rows.append(
            {
                "id": milvus_id,
                "text": chunk.text,
                "repo": ctx["repo"],
                "branch": ctx["branch"],
                "image_tag": ctx["image_tag"],
                "index_version": ctx["index_version"],
                "git_commit": ctx["git_commit"],
                "path": rel_norm,
                "module": _module_of(rel_norm),
                "language": "java",
                "node_type": chunk.node_type,
                "symbol": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "doc_type": "code",
                "content_hash": content_hash,
            }
        )
        meta_rows.append(
            CodeChunkMeta(
                index_version_id=ctx["index_version_id"],
                milvus_id=milvus_id,
                path=rel_norm,
                language="java",
                node_type=chunk.node_type,
                symbol=chunk.name or "",
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content_hash=content_hash,
                char_len=len(chunk.text),
            )
        )

    upserted = milvus_service.upsert_chunks(milvus_rows, settings=settings)

    factory = get_session_factory()
    session = factory()
    try:
        # 同一文件在该 index_version 下先删后插，保持幂等
        old = session.scalars(
            select(CodeChunkMeta).where(
                CodeChunkMeta.index_version_id == ctx["index_version_id"],
                CodeChunkMeta.path == rel_norm,
            )
        ).all()
        for row in old:
            session.delete(row)
        session.flush()
        session.add_all(meta_rows)
        session.flush()

        version = session.get(CodeIndexVersion, ctx["index_version_id"])
        if version is None:
            raise RuntimeError(f"code_index_version id={ctx['index_version_id']} 不存在")

        total_chunks = session.scalar(
            select(func.count())
            .select_from(CodeChunkMeta)
            .where(CodeChunkMeta.index_version_id == ctx["index_version_id"])
        ) or 0
        total_files = session.scalar(
            select(func.count(func.distinct(CodeChunkMeta.path))).where(
                CodeChunkMeta.index_version_id == ctx["index_version_id"]
            )
        ) or 0

        version.chunk_count = int(total_chunks)
        version.file_count = int(total_files)
        version.embedding_model = settings.ollama_embedding_model
        version.embedding_dim = settings.milvus_embedding_dim
        version.milvus_collection = settings.milvus_collection
        version.status = "ready"
        version.ready_at = datetime.now()

        session.commit()
        db_meta = {
            "index_version_id": version.id,
            "meta_upserted": len(meta_rows),
            "file_count": version.file_count,
            "chunk_count": version.chunk_count,
            "status": version.status,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return {
        "file": str(file_path),
        "relative_path": rel_norm,
        "chunk_count": len(chunks),
        "upserted": upserted,
        "db": db_meta,
        "embedding_provider": settings.milvus_embedding_provider,
        "embedding_model": settings.ollama_embedding_model,
        "embedding_dim": settings.milvus_embedding_dim,
        "version": {
            "repo": ctx["repo"],
            "branch": ctx["branch"],
            "image_tag": ctx["image_tag"],
            "index_version": ctx["index_version"],
            "git_commit": ctx["git_commit"],
        },
        "chunks": [
            {
                "id": r["id"],
                "node_type": r["node_type"],
                "symbol": r["symbol"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
            }
            for r in milvus_rows
        ],
    }
