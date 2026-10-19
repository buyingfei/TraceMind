"""Milvus 向量库封装（本地 Docker standalone）。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pymilvus import DataType, MilvusClient

from app.core.config import Settings, get_settings
from app.services.embedding_service import embed_query, embed_texts

# 字段长度按 Milvus varchar 限制预留
_MAX_TEXT = 8192
_MAX_PATH = 1024
_MAX_REPO = 256


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
        # 本地刚启动时 etcd/milvus 可能较慢
        "timeout": 30,
    }
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return MilvusClient(**kwargs)


def health_check(*, settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    client = get_milvus_client()
    collections = client.list_collections()
    exists = cfg.milvus_collection in collections
    row_count = None
    if exists:
        stats = client.get_collection_stats(cfg.milvus_collection)
        row_count = stats.get("row_count")
    return {
        "ok": True,
        "uri": cfg.milvus_uri,
        "collection": cfg.milvus_collection,
        "collection_exists": exists,
        "row_count": row_count,
        "collections": collections,
        "embedding_provider": cfg.milvus_embedding_provider,
        "embedding_dim": cfg.milvus_embedding_dim,
    }


def ensure_collection(*, settings: Settings | None = None, drop_if_exists: bool = False) -> str:
    """创建代码片段集合（若不存在）。"""
    cfg = settings or get_settings()
    client = get_milvus_client()
    name = cfg.milvus_collection

    if drop_if_exists and client.has_collection(name):
        client.drop_collection(name)

    if client.has_collection(name):
        return name

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=cfg.milvus_embedding_dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=_MAX_TEXT)
    schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=_MAX_PATH)
    schema.add_field(field_name="repo", datatype=DataType.VARCHAR, max_length=_MAX_REPO)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    return name


def _clip(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def upsert_chunks(
    chunks: list[dict[str, str]],
    *,
    settings: Settings | None = None,
) -> int:
    """写入/更新文本块。

    chunks 元素需含: id, text；可选 path、repo。
    """
    if not chunks:
        return 0
    cfg = settings or get_settings()
    ensure_collection(settings=cfg)
    client = get_milvus_client()

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, settings=cfg)

    rows = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        rows.append(
            {
                "id": chunk["id"],
                "embedding": vector,
                "text": _clip(chunk.get("text", ""), _MAX_TEXT),
                "path": _clip(chunk.get("path", ""), _MAX_PATH),
                "repo": _clip(chunk.get("repo", ""), _MAX_REPO),
            }
        )

    client.upsert(collection_name=cfg.milvus_collection, data=rows)
    return len(rows)


def search(
    query: str,
    *,
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    cfg = settings or get_settings()
    client = get_milvus_client()
    if not client.has_collection(cfg.milvus_collection):
        return []

    vector = embed_query(query, settings=cfg)
    results = client.search(
        collection_name=cfg.milvus_collection,
        data=[vector],
        limit=top_k,
        output_fields=["text", "path", "repo"],
        search_params={"metric_type": "COSINE"},
    )

    hits: list[dict[str, Any]] = []
    for hit in results[0]:
        entity = hit.get("entity") or {}
        hits.append(
            {
                "id": hit.get("id"),
                "score": hit.get("distance"),
                "text": entity.get("text"),
                "path": entity.get("path"),
                "repo": entity.get("repo"),
            }
        )
    return hits
