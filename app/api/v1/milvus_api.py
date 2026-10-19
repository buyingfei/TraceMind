"""Milvus 联调接口。"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services import milvus_service

router = APIRouter(prefix="/milvus", tags=["milvus"])


class EnsureCollectionRequest(BaseModel):
    drop_if_exists: bool = Field(default=False, description="为 true 时重建集合（会清空数据）")


class UpsertChunk(BaseModel):
    id: str
    text: str
    path: str = ""
    repo: str = ""


class UpsertRequest(BaseModel):
    chunks: list[UpsertChunk]


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


@router.get("/health")
def milvus_health() -> dict:
    try:
        return milvus_service.health_check()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Milvus 不可用: {exc}") from exc


@router.post("/ensure-collection")
def ensure_collection(body: EnsureCollectionRequest | None = None) -> dict:
    body = body or EnsureCollectionRequest()
    try:
        name = milvus_service.ensure_collection(drop_if_exists=body.drop_if_exists)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"创建集合失败: {exc}") from exc
    settings = get_settings()
    return {
        "collection": name,
        "embedding_dim": settings.milvus_embedding_dim,
        "embedding_provider": settings.milvus_embedding_provider,
    }


@router.post("/upsert")
def upsert(body: UpsertRequest) -> dict:
    try:
        count = milvus_service.upsert_chunks([c.model_dump() for c in body.chunks])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"写入失败: {exc}") from exc
    return {"upserted": count}


@router.post("/search")
def search(body: SearchRequest) -> dict:
    try:
        hits = milvus_service.search(body.query, top_k=body.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}") from exc
    return {"query": body.query, "hits": hits}
