"""Milvus 向量库封装（本地 Docker standalone）。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pymilvus import MilvusClient

from app.core.config import Settings, get_settings
from app.db.milvus_models import (
    MAX_PATH,
    MAX_TEXT,
    OUTPUT_FIELDS,
    WT_CODE_CHUNKS,
    empty_code_chunk_row,
    ensure_code_chunks_collection,
)
from app.services.embedding_service import embed_query, embed_texts


def _validate_docker_uri(uri: str) -> str:
    """仅支持 Docker / 远程 Milvus HTTP(S) 地址。"""
    normalized = uri.strip()
    if normalized.startswith(("http://", "https://", "tcp://")):
        return normalized.rstrip("/")
    raise ValueError(
        f"DIAGNOSE_MILVUS_URI 必须是 Docker Milvus 地址，例如 http://127.0.0.1:19530，当前={uri!r}"
    )


@lru_cache
def get_milvus_client() -> MilvusClient:
    settings = get_settings()
    uri = _validate_docker_uri(settings.milvus_uri)
    kwargs: dict[str, Any] = {
        "uri": uri,
        "timeout": 30,
    }
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return MilvusClient(**kwargs)


def health_check(*, settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    client = get_milvus_client()
    collections = client.list_collections()
    name = cfg.milvus_collection
    exists = name in collections
    row_count = None
    if exists:
        stats = client.get_collection_stats(name)
        row_count = stats.get("row_count")
    return {
        "ok": True,
        "uri": cfg.milvus_uri,
        "collection": name,
        "collection_exists": exists,
        "row_count": row_count,
        "collections": collections,
        "embedding_provider": cfg.milvus_embedding_provider,
        "embedding_dim": cfg.milvus_embedding_dim,
    }


def ensure_collection(*, settings: Settings | None = None, drop_if_exists: bool = False) -> str:
    """按 wt_code_chunks 模型创建集合。"""
    cfg = settings or get_settings()
    client = get_milvus_client()
    return ensure_code_chunks_collection(
        client,
        collection_name=cfg.milvus_collection or WT_CODE_CHUNKS,
        embedding_dim=cfg.milvus_embedding_dim,
        drop_if_exists=drop_if_exists,
    )


def _clip(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def get_existing_ids(
    ids: list[str],
    *,
    settings: Settings | None = None,
    batch_size: int = 100,
) -> set[str]:
    """查询哪些 id 已存在于当前 collection。"""
    if not ids:
        return set()
    cfg = settings or get_settings()
    client = get_milvus_client()
    if not client.has_collection(cfg.milvus_collection):
        return set()

    found: set[str] = set()
    step = max(1, batch_size)
    for i in range(0, len(ids), step):
        chunk = [str(x) for x in ids[i : i + step]]
        rows = client.get(collection_name=cfg.milvus_collection, ids=chunk) or []
        for row in rows:
            rid = row.get("id")
            if rid is not None:
                found.add(str(rid))
    return found


def upsert_chunks(
    chunks: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> int:
    """写入/更新代码块。

    每条至少含: id, text；建议带齐 repo/branch/image_tag/index_version/path 等。
    """
    if not chunks:
        return 0
    cfg = settings or get_settings()
    ensure_collection(settings=cfg)
    client = get_milvus_client()

    texts = [str(c.get("text", "")) for c in chunks]
    vectors = embed_texts(texts, settings=cfg)

    rows: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        row = empty_code_chunk_row()
        row.update({k: v for k, v in chunk.items() if k in row and k != "embedding"})
        row["id"] = str(chunk["id"])
        row["embedding"] = vector
        row["text"] = _clip(str(row.get("text") or ""), MAX_TEXT)
        row["path"] = _clip(str(row.get("path") or ""), MAX_PATH)
        row["start_line"] = int(row.get("start_line") or 0)
        row["end_line"] = int(row.get("end_line") or 0)
        if not row.get("doc_type"):
            row["doc_type"] = "code"
        if not row.get("language"):
            row["language"] = "java"
        rows.append(row)

    client.upsert(collection_name=cfg.milvus_collection, data=rows)
    return len(rows)


def search(
    query: str,
    *,
    top_k: int = 5,
    image_tag: str | None = None,
    index_version: str | None = None,
    repo: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    cfg = settings or get_settings()
    client = get_milvus_client()
    if not client.has_collection(cfg.milvus_collection):
        return []

    filters: list[str] = ['doc_type == "code"']
    if repo:
        filters.append(f'repo == "{repo}"')
    if image_tag:
        filters.append(f'image_tag == "{image_tag}"')
    if index_version:
        filters.append(f'index_version == "{index_version}"')
    filter_expr = " and ".join(filters)

    vector = embed_query(query, settings=cfg)
    results = client.search(
        collection_name=cfg.milvus_collection,
        data=[vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=OUTPUT_FIELDS,
        search_params={"metric_type": "COSINE"},
    )

    hits: list[dict[str, Any]] = []
    for hit in results[0]:
        entity = hit.get("entity") or {}
        item = {"id": hit.get("id"), "score": hit.get("distance")}
        item.update({k: entity.get(k) for k in OUTPUT_FIELDS})
        hits.append(item)
    return hits
