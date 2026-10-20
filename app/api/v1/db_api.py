"""数据库联调接口（连通性 / 配置查看）。"""

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.db.session import ping_database

router = APIRouter(prefix="/db", tags=["db"])


@router.get("/health")
def db_health() -> dict:
    try:
        return ping_database()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"MySQL 不可用: {exc}") from exc


@router.get("/config")
def db_config() -> dict:
    """查看连接配置（不返回密码）。"""
    settings = get_settings()
    return {
        "host": settings.mysql_host,
        "port": settings.mysql_port,
        "user": settings.mysql_user,
        "database": settings.mysql_database,
        "charset": settings.mysql_charset,
        "url_masked": (
            f"mysql+pymysql://{settings.mysql_user}:***"
            f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
        ),
    }
