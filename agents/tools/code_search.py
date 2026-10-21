"""代码关键词/语义检索。"""

from __future__ import annotations

import logging
from typing import Any

from app.core.exceptions import NotImplementedCapabilityError
from app.services import milvus_service

logger = logging.getLogger(__name__)

_DEFAULT_REPO = "wintogether-bff"


def search_by_keyword(pattern: str, root: str | None = None) -> list[dict]:
    """后续接入 ripgrep。"""
    raise NotImplementedCapabilityError("tools.code_search.search_by_keyword")


def search_by_embedding(
    query: str,
    top_k: int = 8,
    *,
    repo: str | None = None,
    image_tag: str | None = None,
    index_version: str | None = None,
) -> list[dict]:
    """基于 Milvus 的语义检索（可带索引版本过滤）。"""
    return milvus_service.search(
        query,
        top_k=top_k,
        repo=repo,
        image_tag=image_tag,
        index_version=index_version,
    )


def _load_index_filters(repo_code: str = _DEFAULT_REPO) -> dict[str, Any]:
    """加载当前索引上下文；失败时退化为不按版本过滤。"""
    try:
        from app.services.code_index_service import load_current_index_context

        ctx = load_current_index_context(repo_code)
        return {
            "repo": ctx.get("repo") or repo_code,
            "image_tag": ctx.get("image_tag") or None,
            "index_version": ctx.get("index_version") or None,
            "branch": ctx.get("branch") or "",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_current_index_context 失败，将不做版本过滤: %s", exc)
        return {"repo": repo_code, "image_tag": None, "index_version": None, "branch": ""}


def _hit_key(hit: dict[str, Any]) -> str:
    hid = hit.get("id")
    if hid is not None:
        return str(hid)
    path = hit.get("path") or ""
    start = hit.get("start_line")
    end = hit.get("end_line")
    return f"{path}:{start}-{end}"


def _merge_hits(batches: list[list[dict[str, Any]]], *, limit: int) -> list[dict[str, Any]]:
    """按 id 去重，按 score 降序取前 limit（score 越大越相似，COSINE distance 视 milvus 返回而定）。"""
    best: dict[str, dict[str, Any]] = {}
    for batch in batches:
        for hit in batch or []:
            key = _hit_key(hit)
            prev = best.get(key)
            if prev is None:
                best[key] = hit
                continue
            # 保留更高分（COSINE 下 distance 常为相似度）
            if float(hit.get("score") or 0) > float(prev.get("score") or 0):
                best[key] = hit
    merged = sorted(best.values(), key=lambda h: float(h.get("score") or 0), reverse=True)
    return merged[: max(1, limit)]


def retrieve_for_diagnose(
    queries: list[str],
    *,
    top_k_per_query: int = 6,
    final_top_k: int = 8,
    repo: str = _DEFAULT_REPO,
) -> dict[str, Any]:
    """多路语义检索并合并；自动带上当前索引版本过滤。"""
    filters = _load_index_filters(repo)
    cleaned = [q.strip() for q in queries if (q or "").strip()]
    if not cleaned:
        return {
            "hits": [],
            "queries": [],
            "index": filters,
            "error": "检索 query 为空",
        }

    batches: list[list[dict[str, Any]]] = []
    errors: list[str] = []
    for q in cleaned:
        try:
            hits = search_by_embedding(
                q,
                top_k=top_k_per_query,
                repo=filters.get("repo"),
                image_tag=filters.get("image_tag"),
                index_version=filters.get("index_version"),
            )
            batches.append(hits)
        except Exception as exc:  # noqa: BLE001
            logger.warning("retrieve query 失败 q=%s: %s", q[:80], exc)
            errors.append(f"{q[:40]}: {exc}")
            batches.append([])

    hits = _merge_hits(batches, limit=final_top_k)
    return {
        "hits": hits,
        "queries": cleaned,
        "index": filters,
        "error": " | ".join(errors) if errors else "",
    }
