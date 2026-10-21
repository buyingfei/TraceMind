"""FastAPI 应用入口。"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        from app.services.checkpoint_service import ensure_checkpoint_ready

        info = ensure_checkpoint_ready()
        logger.info("checkpoint ready: %s", info)
    except Exception:  # noqa: BLE001
        logger.exception("LangGraph checkpoint 初始化失败（对话记忆不可用）")
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="业务问题排查 Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "message": "Hello World",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
