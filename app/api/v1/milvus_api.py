"""Milvus 联调 / 索引 / 检索接口。"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services import milvus_service
from app.services.code_index_service import (
    get_index_job,
    index_java_file,
    load_current_index_context,
    start_index_repo,
)

router = APIRouter(prefix="/milvus", tags=["milvus"])


class EnsureCollectionRequest(BaseModel):
    drop_if_exists: bool = Field(default=False, description="为 true 时重建集合（会清空数据）")


class IndexPreviewRequest(BaseModel):
    relative_path: str | None = Field(
        default=None,
        description="相对仓库路径；为空则使用 config/code_index.yaml 的 preview.file",
    )
    recreate_collection: bool = Field(
        default=False,
        description="维度变更时可 true 重建 wt_code_chunks",
    )


class IndexRepoRequest(BaseModel):
    repo: str = Field(..., description="项目名，对应 code_repo.repo_code，如 wintogether-bff")
    index_version: str | None = Field(
        default=None,
        description="传入则续跑该版本（跳过已入库文件并修复缺向量）；不传则新建全量",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="自然语言或代码相关问题")
    top_k: int = Field(default=5, ge=1, le=50)
    image_tag: str | None = Field(default=None, description="不传则用 MySQL 当前 is_current 版本")
    index_version: str | None = None
    repo: str | None = Field(default="wintogether-bff")


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
        "model": "wt_code_chunks",
        "embedding_dim": settings.milvus_embedding_dim,
        "embedding_provider": settings.milvus_embedding_provider,
    }


@router.post("/index/preview-file")
def index_preview_file(body: IndexPreviewRequest | None = None) -> dict:
    """切块 + 先写 MySQL meta + 再写 Milvus（当前 is_current 版本）。"""
    body = body or IndexPreviewRequest()
    try:
        return index_java_file(
            body.relative_path,
            recreate_collection=body.recreate_collection,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"索引失败: {exc}") from exc


@router.post("/index/repo")
def index_repo(body: IndexRepoRequest) -> dict:
    """全量或续跑索引。传 index_version 则从断点续跑，否则新建版本全量。"""
    try:
        return start_index_repo(
            body.repo.strip(),
            index_version=(body.index_version.strip() if body.index_version else None),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"提交全量索引失败: {exc}") from exc


@router.get("/index/repo/{index_version}")
def index_repo_status(index_version: str) -> dict:
    """查询全量索引任务进度。"""
    job = get_index_job(index_version)
    if job is None:
        raise HTTPException(status_code=404, detail=f"未找到任务: {index_version}")
    return job


@router.post("/search")
def search(body: SearchRequest) -> dict:
    """按问题检索代码块（默认过滤当前 image_tag）。"""
    try:
        image_tag = body.image_tag
        index_version = body.index_version
        repo = body.repo or "wintogether-bff"
        if not image_tag:
            ctx = load_current_index_context(repo)
            image_tag = ctx["image_tag"]
            if not index_version:
                index_version = ctx["index_version"]
            repo = ctx["repo"]

        hits = milvus_service.search(
            body.query,
            top_k=body.top_k,
            image_tag=image_tag,
            index_version=index_version,
            repo=repo,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"检索失败: {exc}") from exc

    return {
        "query": body.query,
        "repo": repo,
        "image_tag": image_tag,
        "index_version": index_version,
        "hit_count": len(hits),
        "hits": hits,
    }
