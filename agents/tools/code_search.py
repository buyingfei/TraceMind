"""代码关键词/语义检索。"""

from __future__ import annotations

from app.core.exceptions import NotImplementedCapabilityError
from app.services import milvus_service


def search_by_keyword(pattern: str, root: str | None = None) -> list[dict]:
    """后续接入 ripgrep。"""
    raise NotImplementedCapabilityError("tools.code_search.search_by_keyword")


def search_by_embedding(query: str, top_k: int = 8) -> list[dict]:
    """基于 Milvus 的语义检索。"""
    return milvus_service.search(query, top_k=top_k)
