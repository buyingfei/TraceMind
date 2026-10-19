"""读取源码片段（骨架占位）。"""

from app.core.exceptions import NotImplementedCapabilityError


def read_snippet(path: str, start_line: int = 1, end_line: int = 80) -> str:
    """后续按路径读取代码仓库中的文件片段。"""
    raise NotImplementedCapabilityError("tools.file_reader.read_snippet")
