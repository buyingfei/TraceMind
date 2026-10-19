"""重排序服务（骨架占位）。"""

from app.core.exceptions import NotImplementedCapabilityError


def rerank(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """后续接入 DashScope Rerank。"""
    raise NotImplementedCapabilityError("rerank_service.rerank")
