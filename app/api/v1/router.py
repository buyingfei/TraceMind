"""API v1 路由汇总。"""

from fastapi import APIRouter

from app.api.v1 import hello, llm, tickets

api_router = APIRouter()
api_router.include_router(hello.router, tags=["hello"])
api_router.include_router(llm.router, tags=["llm"])
api_router.include_router(tickets.router, prefix="/tickets", tags=["tickets"])
