"""文本向量化。

- ollama: 本地 Ollama Embedding（如 batiai/qwen3-embedding:0.6b）
- openai: OpenAI 兼容网关 Embedding
- local_hash: 伪向量，仅测连通
"""

from __future__ import annotations

import hashlib
import math
import struct

import httpx

from app.core.config import Settings, get_settings
from app.infra.llm import get_openai_client


def _local_hash_embedding(text: str, dim: int) -> list[float]:
    values: list[float] = []
    seed = text.encode("utf-8")
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
        for i in range(0, len(digest), 4):
            if len(values) >= dim:
                break
            (num,) = struct.unpack("I", digest[i : i + 4])
            values.append((num / 0xFFFFFFFF) * 2.0 - 1.0)
        counter += 1
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _embed_ollama(texts: list[str], settings: Settings) -> list[list[float]]:
    url = settings.ollama_base_url.rstrip("/") + "/api/embed"
    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            url,
            json={
                "model": settings.ollama_embedding_model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
    vectors = data.get("embeddings") or []
    if len(vectors) != len(texts):
        raise RuntimeError(f"Ollama 返回向量数 {len(vectors)} 与输入 {len(texts)} 不一致")
    return [list(v) for v in vectors]


def _embed_openai(texts: list[str], settings: Settings) -> list[list[float]]:
    client = get_openai_client()
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    sorted_data = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in sorted_data]


def embed_texts(texts: list[str], *, settings: Settings | None = None) -> list[list[float]]:
    if not texts:
        return []
    cfg = settings or get_settings()
    provider = (cfg.milvus_embedding_provider or "openai").lower()

    if provider == "local_hash":
        vectors = [_local_hash_embedding(t, cfg.milvus_embedding_dim) for t in texts]
    elif provider == "ollama":
        vectors = _embed_ollama(texts, cfg)
    elif provider == "openai":
        vectors = _embed_openai(texts, cfg)
    else:
        raise ValueError(f"不支持的 embedding provider: {provider}")

    if vectors and len(vectors[0]) != cfg.milvus_embedding_dim:
        raise ValueError(
            f"Embedding 维度 {len(vectors[0])} 与配置 MILVUS_EMBEDDING_DIM="
            f"{cfg.milvus_embedding_dim} 不一致，请修改 .env 或重建 Milvus collection"
        )
    return vectors


def embed_query(text: str, *, settings: Settings | None = None) -> list[float]:
    return embed_texts([text], settings=settings)[0]
