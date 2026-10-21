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


_CRITICAL_SYMBOLS = {
    "isFalseClaim",
    "isChainStore",
    "createOut",
    "createIn",
    "k2InAndOrgCodeNotEquals",
    "k2OutAndOrgCodeNotEquals",
    "queryScanStoreTyre",
}


def _is_critical_hit(hit: dict[str, Any]) -> bool:
    symbol = str(hit.get("symbol") or "")
    if symbol in _CRITICAL_SYMBOLS:
        return True
    path = str(hit.get("path") or "")
    text = str(hit.get("text") or "")
    if "RetailCommService" in path and "isFalseClaim" in (symbol + text[:200]):
        return True
    return any(name in symbol for name in _CRITICAL_SYMBOLS)


def _merge_hits(
    batches: list[list[dict[str, Any]]],
    *,
    limit: int,
    prefer_critical: bool = False,
) -> list[dict[str, Any]]:
    """按 id 去重，按 score 降序取前 limit；prefer_critical 时优先保留关键路径命中。"""
    best: dict[str, dict[str, Any]] = {}
    for batch in batches:
        for hit in batch or []:
            key = _hit_key(hit)
            prev = best.get(key)
            if prev is None:
                best[key] = hit
                continue
            if float(hit.get("score") or 0) > float(prev.get("score") or 0):
                best[key] = hit

    all_hits = list(best.values())
    ranked = sorted(all_hits, key=lambda h: float(h.get("score") or 0), reverse=True)
    if not prefer_critical:
        return ranked[: max(1, limit)]

    critical = [h for h in ranked if _is_critical_hit(h)]
    others = [h for h in ranked if not _is_critical_hit(h)]
    # 关键路径至少占一半名额（有则优先），其余按分数补齐
    reserve = min(len(critical), max(3, limit // 2))
    picked = critical[:reserve]
    picked_keys = {_hit_key(h) for h in picked}
    for h in others:
        if len(picked) >= limit:
            break
        if _hit_key(h) in picked_keys:
            continue
        picked.append(h)
        picked_keys.add(_hit_key(h))
    # 若名额仍空，继续塞剩余 critical
    for h in critical[reserve:]:
        if len(picked) >= limit:
            break
        if _hit_key(h) in picked_keys:
            continue
        picked.append(h)
    return picked[: max(1, limit)]


def retrieve_for_diagnose(
    queries: list[str],
    *,
    top_k_per_query: int = 6,
    final_top_k: int = 8,
    repo: str = _DEFAULT_REPO,
    prefer_critical: bool = False,
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

    hits = _merge_hits(
        batches,
        limit=final_top_k,
        prefer_critical=prefer_critical,
    )
    return {
        "hits": hits,
        "queries": cleaned,
        "index": filters,
        "error": " | ".join(errors) if errors else "",
    }
