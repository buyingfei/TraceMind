"""基础设施：DB / LLM / OCR / 业务只读库等客户端工厂。"""

from app.infra.biz_db import biz_readonly_enabled, get_biz_readonly_engine
from app.infra.db import get_engine, get_session, get_session_factory, ping_database
from app.infra.llm import client_for_provider, get_ollama_openai_client, get_openai_client
from app.infra.ocr import get_paddle_ocr

__all__ = [
    "biz_readonly_enabled",
    "client_for_provider",
    "get_biz_readonly_engine",
    "get_engine",
    "get_ollama_openai_client",
    "get_openai_client",
    "get_paddle_ocr",
    "get_session",
    "get_session_factory",
    "ping_database",
]
