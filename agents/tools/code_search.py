"""代码关键词/语义检索（骨架占位）。"""

from app.core.exceptions import NotImplementedCapabilityError


def search_by_keyword(pattern: str, root: str | None = None) -> list[dict]:
    """后续接入 ripgrep。"""
    raise NotImplementedCapabilityError("tools.code_search.search_by_keyword")


def search_by_embedding(query: str, top_k: int = 8) -> list[dict]:
    """后续接入 Milvus。"""
    raise NotImplementedCapabilityError("tools.code_search.search_by_embedding")
