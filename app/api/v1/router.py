"""API v1 路由汇总。"""

from fastapi import APIRouter

from app.api.v1 import code_split, db_api, hello, llm, milvus_api, tickets

api_router = APIRouter()
api_router.include_router(hello.router, tags=["hello"])
api_router.include_router(llm.router, tags=["llm"])
api_router.include_router(milvus_api.router)
api_router.include_router(code_split.router)
api_router.include_router(db_api.router)
api_router.include_router(tickets.router, prefix="/tickets", tags=["tickets"])
