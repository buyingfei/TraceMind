"""日志初始化；文本摘要工具见 app.common.text。"""

from __future__ import annotations

import logging

from app.common.text import clip_text, summarize_messages

__all__ = ["clip_text", "setup_logging", "summarize_messages"]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
