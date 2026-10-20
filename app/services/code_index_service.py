"""将 Java 源码切块、向量化并写入 MySQL meta / Milvus。

写入顺序：每批 chunk 先落 MySQL code_chunk_meta，再 upsert Milvus。
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)

_job_lock = threading.Lock()
_running_repos: set[str] = set()
# index_version -> 进度快照（内存；权威状态仍以 MySQL code_index_version.status 为准）
_jobs: dict[str, dict[str, Any]] = {}


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


def _read_git_commit(local_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(local_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            return (result.stdout or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 git commit 失败: %s", exc)
    return ""


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


def _build_chunk_rows(
    *,
    chunks: list[Any],
    ctx: dict[str, Any],
    rel_norm: str,
) -> tuple[list[dict[str, Any]], list[CodeChunkMeta]]:
    milvus_rows: list[dict[str, Any]] = []
    meta_rows: list[CodeChunkMeta] = []
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
    return milvus_rows, meta_rows


def _persist_batch_mysql_then_milvus(
    *,
    milvus_rows: list[dict[str, Any]],
    meta_rows: list[CodeChunkMeta],
    replace_path: str | None = None,
    index_version_id: int | None = None,
) -> int:
    """先写 MySQL，再写 Milvus。返回 upsert 条数。"""
    if not milvus_rows:
        return 0

    settings = get_settings()
    factory = get_session_factory()
    session = factory()
    try:
        if replace_path is not None and index_version_id is not None:
            old = session.scalars(
                select(CodeChunkMeta).where(
                    CodeChunkMeta.index_version_id == index_version_id,
                    CodeChunkMeta.path == replace_path,
                )
            ).all()
            for row in old:
                session.delete(row)
            session.flush()
        session.add_all(meta_rows)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return milvus_service.upsert_chunks(milvus_rows, settings=settings)


def _refresh_version_counts(index_version_id: int, *, status: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    factory = get_session_factory()
    session = factory()
    try:
        version = session.get(CodeIndexVersion, index_version_id)
        if version is None:
            raise RuntimeError(f"code_index_version id={index_version_id} 不存在")

        total_chunks = session.scalar(
            select(func.count())
            .select_from(CodeChunkMeta)
            .where(CodeChunkMeta.index_version_id == index_version_id)
        ) or 0
        total_files = session.scalar(
            select(func.count(func.distinct(CodeChunkMeta.path))).where(
                CodeChunkMeta.index_version_id == index_version_id
            )
        ) or 0

        version.chunk_count = int(total_chunks)
        version.file_count = int(total_files)
        version.embedding_model = settings.ollama_embedding_model
        version.embedding_dim = settings.milvus_embedding_dim
        version.milvus_collection = settings.milvus_collection
        if status:
            version.status = status
            if status == "ready":
                version.ready_at = datetime.now()
        session.commit()
        return {
            "index_version_id": version.id,
            "file_count": version.file_count,
            "chunk_count": version.chunk_count,
            "status": version.status,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def index_java_file(
    relative_path: str | None = None,
    *,
    recreate_collection: bool = False,
) -> dict[str, Any]:
    """切块指定 Java 文件 → 先 MySQL meta → 再 Milvus。"""
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

    rel_norm = rel.replace("\\", "/")
    milvus_rows, meta_rows = _build_chunk_rows(chunks=chunks, ctx=ctx, rel_norm=rel_norm)
    upserted = _persist_batch_mysql_then_milvus(
        milvus_rows=milvus_rows,
        meta_rows=meta_rows,
        replace_path=rel_norm,
        index_version_id=ctx["index_version_id"],
    )
    db_meta = _refresh_version_counts(ctx["index_version_id"], status="ready")
    db_meta["meta_upserted"] = len(meta_rows)

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


def _scan_java_files(local_root: Path, scan_cfg: dict[str, Any]) -> list[Path]:
    include_globs = scan_cfg.get("include_globs") or ["**/src/main/java/**/*.java"]
    exclude_dirs = {str(d).lower() for d in (scan_cfg.get("exclude_dirs") or [])}
    found: list[Path] = []
    seen: set[Path] = set()

    for pattern in include_globs:
        for path in local_root.glob(pattern):
            if not path.is_file():
                continue
            try:
                rel_parts = path.relative_to(local_root).parts
            except ValueError:
                continue
            if any(part.lower() in exclude_dirs for part in rel_parts):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)

    found.sort(key=lambda p: str(p).lower())
    return found


def _new_index_version_code() -> str:
    return datetime.now().strftime("idx_%Y%m%d_%H%M%S")


def _set_job(index_version: str, **kwargs: Any) -> None:
    with _job_lock:
        job = _jobs.setdefault(index_version, {})
        job.update(kwargs)
        job["updated_at"] = datetime.now().isoformat(timespec="seconds")


def get_index_job(index_version: str) -> dict[str, Any] | None:
    """查询全量任务进度（内存 + MySQL status）。"""
    with _job_lock:
        snapshot = dict(_jobs.get(index_version) or {})

    factory = get_session_factory()
    session = factory()
    try:
        version = session.scalar(
            select(CodeIndexVersion).where(CodeIndexVersion.index_version == index_version)
        )
        if version is None and not snapshot:
            return None
        if version is not None:
            snapshot.setdefault("index_version", version.index_version)
            snapshot.setdefault("index_version_id", version.id)
            snapshot["status"] = version.status
            snapshot["file_count"] = version.file_count
            snapshot["chunk_count"] = version.chunk_count
            snapshot["branch"] = version.branch
            snapshot["image_tag"] = version.image_tag
            repo = session.get(CodeRepo, version.repo_id)
            if repo:
                snapshot["repo"] = repo.repo_code
        return snapshot or None
    finally:
        session.close()


def _mark_version_failed(index_version_id: int, remark: str) -> None:
    factory = get_session_factory()
    session = factory()
    try:
        version = session.get(CodeIndexVersion, index_version_id)
        if version is None:
            return
        version.status = "failed"
        version.remark = (remark or "")[:500]
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("标记 index_version failed 失败")
    finally:
        session.close()


def _switch_is_current(repo_id: int, new_version_id: int) -> None:
    factory = get_session_factory()
    session = factory()
    try:
        versions = session.scalars(
            select(CodeIndexVersion).where(CodeIndexVersion.repo_id == repo_id)
        ).all()
        for ver in versions:
            ver.is_current = 1 if ver.id == new_version_id else 0
        target = session.get(CodeIndexVersion, new_version_id)
        if target is None:
            raise RuntimeError(f"切换 is_current 失败: id={new_version_id}")
        target.status = "ready"
        target.ready_at = datetime.now()
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _meta_paths_and_ids(index_version_id: int) -> tuple[set[str], dict[str, list[str]]]:
    """返回已有 meta 的 path 集合，以及 path -> milvus_id 列表。"""
    factory = get_session_factory()
    session = factory()
    try:
        rows = session.scalars(
            select(CodeChunkMeta).where(CodeChunkMeta.index_version_id == index_version_id)
        ).all()
        path_ids: dict[str, list[str]] = {}
        for row in rows:
            path_ids.setdefault(row.path, []).append(row.milvus_id)
        return set(path_ids.keys()), path_ids
    finally:
        session.close()


def _paths_missing_in_milvus(
    path_ids: dict[str, list[str]],
    *,
    settings: Any,
) -> set[str]:
    """meta 有但 Milvus 缺向量的 path。"""
    all_ids: list[str] = []
    for ids in path_ids.values():
        all_ids.extend(ids)
    existing = milvus_service.get_existing_ids(all_ids, settings=settings)
    missing_paths: set[str] = set()
    for path, ids in path_ids.items():
        if not ids or any(mid not in existing for mid in ids):
            missing_paths.add(path)
    return missing_paths


def _run_index_repo(job: dict[str, Any]) -> None:
    repo_code = job["repo"]
    index_version = job["index_version"]
    index_version_id = job["index_version_id"]
    mode = job.get("mode") or "full"
    local_root = Path(job["local_root"])
    cfg = load_code_index_config()
    scan_cfg = cfg.get("scan") or {}
    splitter_cfg = cfg.get("splitter") or {}
    batch_size = max(1, int(scan_cfg.get("batch_size") or 16))
    settings = get_settings()

    ctx = {
        "repo": repo_code,
        "branch": job["branch"],
        "image_tag": job["image_tag"],
        "git_commit": job["git_commit"],
        "index_version": index_version,
        "index_version_id": index_version_id,
    }

    try:
        milvus_service.ensure_collection(settings=settings, drop_if_exists=False)
        files = _scan_java_files(local_root, scan_cfg)
        total_files = len(files)

        done_paths, path_ids = _meta_paths_and_ids(index_version_id)
        repair_paths: set[str] = set()
        if mode == "resume" and path_ids:
            _set_job(
                index_version,
                mode=mode,
                message="续跑：检查 MySQL/Milvus 一致性…",
            )
            repair_paths = _paths_missing_in_milvus(path_ids, settings=settings)

        skip_paths = done_paths - repair_paths
        repair_files = [
            p
            for p in files
            if p.relative_to(local_root).as_posix() in repair_paths
        ]
        new_files = [
            p
            for p in files
            if p.relative_to(local_root).as_posix() not in done_paths
        ]
        remaining_count = len(repair_files) + len(new_files)

        _set_job(
            index_version,
            status="building",
            mode=mode,
            total_files=total_files,
            skipped_files=len(skip_paths),
            remaining_files=remaining_count,
            repaired_files=0,
            processed_files=0,
            chunk_count=0,
            failed_files=[],
            message=(
                f"扫描 {total_files}；跳过 {len(skip_paths)}；"
                f"修复 {len(repair_files)}；新增 {len(new_files)}"
            ),
        )

        pending_milvus: list[dict[str, Any]] = []
        pending_meta: list[CodeChunkMeta] = []
        processed = 0
        repaired = 0
        chunk_total = 0
        failed: list[dict[str, str]] = []

        def flush_batch(*, milvus_only: bool) -> None:
            nonlocal pending_milvus, pending_meta, chunk_total
            if not pending_milvus:
                return
            if milvus_only:
                n = milvus_service.upsert_chunks(pending_milvus, settings=settings)
            else:
                n = _persist_batch_mysql_then_milvus(
                    milvus_rows=pending_milvus,
                    meta_rows=pending_meta,
                )
            chunk_total += n
            pending_milvus = []
            pending_meta = []
            _refresh_version_counts(index_version_id, status="building")
            _set_job(
                index_version,
                processed_files=processed,
                repaired_files=repaired,
                chunk_count=chunk_total,
                failed_files=failed[-20:],
                message=(
                    f"进度 {processed}/{remaining_count}，"
                    f"写入chunk={chunk_total}，修复文件={repaired}"
                ),
            )

        def index_file_list(
            file_list: list[Path],
            *,
            milvus_only: bool,
        ) -> None:
            nonlocal processed, repaired, pending_milvus, pending_meta
            for file_path in file_list:
                rel_norm = file_path.relative_to(local_root).as_posix()
                try:
                    chunks = split_java_file(
                        file_path,
                        node_types=splitter_cfg.get("node_types"),
                        include_preamble=bool(splitter_cfg.get("include_preamble", True)),
                        max_chars=int(splitter_cfg.get("max_chars", 4000)),
                    )
                    milvus_rows, meta_rows = _build_chunk_rows(
                        chunks=chunks, ctx=ctx, rel_norm=rel_norm
                    )
                    pending_milvus.extend(milvus_rows)
                    if not milvus_only:
                        pending_meta.extend(meta_rows)
                    if milvus_only:
                        repaired += 1
                    if len(pending_milvus) >= batch_size:
                        flush_batch(milvus_only=milvus_only)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("索引文件失败: %s", rel_norm)
                    failed.append({"path": rel_norm, "error": str(exc)[:300]})
                finally:
                    processed += 1
                    if processed % 20 == 0 or processed == remaining_count:
                        _set_job(
                            index_version,
                            processed_files=processed,
                            remaining_files=remaining_count,
                            skipped_files=len(skip_paths),
                            repaired_files=repaired,
                            total_files=total_files,
                            chunk_count=chunk_total + len(pending_milvus),
                            failed_files=failed[-20:],
                            message=(
                                f"进度 {processed}/{remaining_count}；"
                                f"跳过 {len(skip_paths)}；修复 {repaired}"
                            ),
                        )
            flush_batch(milvus_only=milvus_only)

        # 先补齐「有 meta 无向量」，再索引全新文件
        index_file_list(repair_files, milvus_only=True)
        index_file_list(new_files, milvus_only=False)

        counts = _refresh_version_counts(index_version_id, status="building")
        _switch_is_current(job["repo_id"], index_version_id)
        _set_job(
            index_version,
            status="ready",
            mode=mode,
            processed_files=processed,
            total_files=total_files,
            skipped_files=len(skip_paths),
            remaining_files=remaining_count,
            repaired_files=repaired,
            chunk_count=counts["chunk_count"],
            file_count=counts["file_count"],
            failed_files=failed[-50:],
            failed_count=len(failed),
            message="索引完成，已切换 is_current",
        )
        logger.info(
            "索引完成 mode=%s repo=%s index_version=%s files=%s chunks=%s "
            "skipped=%s repaired=%s failed=%s",
            mode,
            repo_code,
            index_version,
            counts["file_count"],
            counts["chunk_count"],
            len(skip_paths),
            repaired,
            len(failed),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("索引失败 repo=%s mode=%s", repo_code, mode)
        _mark_version_failed(index_version_id, str(exc))
        _set_job(
            index_version,
            status="failed",
            mode=mode,
            message=str(exc)[:500],
        )
    finally:
        with _job_lock:
            _running_repos.discard(repo_code)


def start_index_repo(
    repo_code: str,
    index_version: str | None = None,
) -> dict[str, Any]:
    """全量或续跑：不传 index_version 新建；传入则在该版本上跳过已完成文件继续跑。"""
    with _job_lock:
        if repo_code in _running_repos:
            raise RuntimeError(f"项目 {repo_code} 已有全量索引任务在执行")
        _running_repos.add(repo_code)

    settings = get_settings()
    factory = get_session_factory()
    session = factory()
    job: dict[str, Any] | None = None
    mode = "resume" if index_version else "full"
    try:
        repo = session.scalar(select(CodeRepo).where(CodeRepo.repo_code == repo_code))
        if repo is None:
            raise RuntimeError(f"MySQL 中不存在 code_repo: {repo_code}")

        local_root = Path(repo.local_root)
        if not local_root.exists():
            raise FileNotFoundError(f"local_root 不存在: {local_root}")

        if mode == "resume":
            assert index_version is not None
            version = session.scalar(
                select(CodeIndexVersion).where(
                    CodeIndexVersion.repo_id == repo.id,
                    CodeIndexVersion.index_version == index_version,
                )
            )
            if version is None:
                raise RuntimeError(
                    f"未找到 index_version={index_version}（repo={repo_code}）"
                )
            if version.status == "ready":
                raise RuntimeError(
                    f"{index_version} 已是 ready，不能续跑；全量请不要传 index_version"
                )
            if version.status not in ("building", "failed"):
                raise RuntimeError(
                    f"{index_version} 状态={version.status}，仅 building/failed 可续跑"
                )

            other_building = session.scalar(
                select(CodeIndexVersion).where(
                    CodeIndexVersion.repo_id == repo.id,
                    CodeIndexVersion.status == "building",
                    CodeIndexVersion.id != version.id,
                )
            )
            if other_building is not None:
                raise RuntimeError(
                    f"项目 {repo_code} 另有 building 版本: {other_building.index_version}"
                )

            version.status = "building"
            version.remark = f"resume {datetime.now().isoformat(timespec='seconds')}"
            if version.started_at is None:
                version.started_at = datetime.now()
            session.commit()
            session.refresh(version)

            done_paths, _ = _meta_paths_and_ids(version.id)
            job = {
                "repo": repo.repo_code,
                "repo_id": repo.id,
                "local_root": repo.local_root,
                "branch": version.branch,
                "image_tag": version.image_tag,
                "git_commit": version.git_commit,
                "index_version": version.index_version,
                "index_version_id": version.id,
                "mode": "resume",
                "status": "building",
                "skipped_files": len(done_paths),
                "total_files": 0,
                "remaining_files": 0,
                "repaired_files": 0,
                "processed_files": 0,
                "chunk_count": version.chunk_count,
                "file_count": version.file_count,
                "failed_files": [],
                "failed_count": 0,
                "message": f"续跑任务已创建，已有 meta 文件约 {len(done_paths)} 个",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        else:
            building = session.scalar(
                select(CodeIndexVersion).where(
                    CodeIndexVersion.repo_id == repo.id,
                    CodeIndexVersion.status == "building",
                )
            )
            if building is not None:
                raise RuntimeError(
                    f"项目 {repo_code} 已有 building 版本: {building.index_version}；"
                    f"续跑请传 index_version"
                )

            current = session.scalar(
                select(CodeIndexVersion).where(
                    CodeIndexVersion.repo_id == repo.id,
                    CodeIndexVersion.is_current == 1,
                )
            )
            if current is None:
                raise RuntimeError(
                    f"MySQL 中不存在 is_current=1 的版本（repo={repo_code}）"
                )

            git_commit = _read_git_commit(local_root) or current.git_commit
            new_code = _new_index_version_code()
            exists = session.scalar(
                select(CodeIndexVersion).where(
                    CodeIndexVersion.repo_id == repo.id,
                    CodeIndexVersion.index_version == new_code,
                )
            )
            if exists is not None:
                new_code = f"{new_code}_{_md5(new_code)[:6]}"

            version = CodeIndexVersion(
                repo_id=repo.id,
                branch=current.branch,
                image_tag=current.image_tag,
                git_commit=git_commit,
                index_version=new_code,
                embedding_model=settings.ollama_embedding_model,
                embedding_dim=settings.milvus_embedding_dim,
                milvus_collection=settings.milvus_collection,
                status="building",
                file_count=0,
                chunk_count=0,
                is_current=0,
                remark="full repo index",
                started_at=datetime.now(),
            )
            session.add(version)
            session.commit()
            session.refresh(version)

            job = {
                "repo": repo.repo_code,
                "repo_id": repo.id,
                "local_root": repo.local_root,
                "branch": version.branch,
                "image_tag": version.image_tag,
                "git_commit": version.git_commit,
                "index_version": version.index_version,
                "index_version_id": version.id,
                "mode": "full",
                "status": "building",
                "skipped_files": 0,
                "total_files": 0,
                "remaining_files": 0,
                "repaired_files": 0,
                "processed_files": 0,
                "chunk_count": 0,
                "file_count": 0,
                "failed_files": [],
                "failed_count": 0,
                "message": "全量任务已创建，后台执行中",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }

        with _job_lock:
            _jobs[job["index_version"]] = job
    except Exception:
        session.rollback()
        with _job_lock:
            _running_repos.discard(repo_code)
        raise
    finally:
        session.close()

    assert job is not None
    thread = threading.Thread(
        target=_run_index_repo,
        args=(dict(job),),
        name=f"index-repo-{repo_code}-{job['index_version']}",
        daemon=True,
    )
    thread.start()

    return {
        "job_id": job["index_version"],
        "repo": repo_code,
        "mode": job["mode"],
        "branch": job["branch"],
        "image_tag": job["image_tag"],
        "index_version": job["index_version"],
        "index_version_id": job["index_version_id"],
        "git_commit": job["git_commit"],
        "skipped_files": job.get("skipped_files", 0),
        "status": "building",
    }

