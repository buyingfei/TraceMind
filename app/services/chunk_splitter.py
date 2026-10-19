"""文本切块服务（骨架占位）。"""

from app.core.exceptions import NotImplementedCapabilityError


def split_text(text: str, chunk_size: int = 500) -> list[str]:
    """后续用于代码/文档入库前切块。"""
    raise NotImplementedCapabilityError("chunk_splitter.split_text")
