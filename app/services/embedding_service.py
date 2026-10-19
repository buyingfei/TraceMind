"""文本向量化。

- openai: 走 OpenAI 兼容网关 Embedding 接口
- local_hash: 本地确定性伪向量，仅用于验证 Milvus 连通（不适合真实检索）
"""

from __future__ import annotations

import hashlib
import math
import struct

from app.core.config import Settings, get_settings
from app.services.llm_service import get_openai_client


def _local_hash_embedding(text: str, dim: int) -> list[float]:
    """用哈希生成固定维度向量，并做 L2 归一化。"""
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


def embed_texts(texts: list[str], *, settings: Settings | None = None) -> list[list[float]]:
    if not texts:
        return []
    cfg = settings or get_settings()
    provider = (cfg.milvus_embedding_provider or "openai").lower()

    if provider == "local_hash":
        return [_local_hash_embedding(t, cfg.milvus_embedding_dim) for t in texts]

    if provider != "openai":
        raise ValueError(f"不支持的 embedding provider: {provider}")

    client = get_openai_client()
    response = client.embeddings.create(
        model=cfg.openai_embedding_model,
        input=texts,
    )
    # OpenAI 按 index 回传，这里按输入顺序整理
    sorted_data = sorted(response.data, key=lambda item: item.index)
    vectors = [list(item.embedding) for item in sorted_data]
    if vectors and len(vectors[0]) != cfg.milvus_embedding_dim:
        raise ValueError(
            f"Embedding 维度 {len(vectors[0])} 与配置 MILVUS_EMBEDDING_DIM="
            f"{cfg.milvus_embedding_dim} 不一致，请修改 .env"
        )
    return vectors


def embed_query(text: str, *, settings: Settings | None = None) -> list[float]:
    return embed_texts([text], settings=settings)[0]
