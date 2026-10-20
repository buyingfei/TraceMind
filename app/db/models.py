"""业务元数据表模型（MySQL）。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CodeRepo(Base):
    __tablename__ = "code_repo"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    repo_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, comment="仓库编码")
    repo_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="展示名")
    local_root: Mapped[str] = mapped_column(String(512), nullable=False, comment="本地根目录")
    status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1, comment="1启用 0停用")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class CodeIndexVersion(Base):
    __tablename__ = "code_index_version"
    __table_args__ = (
        UniqueConstraint("repo_id", "image_tag", "index_version", name="uk_repo_image_index"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    repo_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True, comment="code_repo.id")
    branch: Mapped[str] = mapped_column(String(128), nullable=False, comment="Git 分支")
    image_tag: Mapped[str] = mapped_column(String(128), nullable=False, comment="镜像号")
    git_commit: Mapped[str] = mapped_column(String(40), nullable=False, default="", comment="commit sha")
    index_version: Mapped[str] = mapped_column(String(64), nullable=False, comment="索引批次号")
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False, comment="Embedding 模型")
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False, comment="向量维度")
    milvus_collection: Mapped[str] = mapped_column(String(128), nullable=False, comment="Milvus collection")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="building", comment="building/ready/deprecated/failed"
    )
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_current: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, comment="是否当前问答版本")
    remark: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())


class CodeChunkMeta(Base):
    """切块元数据（对账/增量）；向量在 Milvus，全文可不落此表。"""

    __tablename__ = "code_chunk_meta"
    __table_args__ = (
        UniqueConstraint("index_version_id", "milvus_id", name="uk_version_milvus_id"),
        Index("idx_chunk_meta_version_path", "index_version_id", "path", mysql_length={"path": 255}),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    index_version_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, index=True, comment="code_index_version.id"
    )
    milvus_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="与完整 Milvus id 一致")
    path: Mapped[str] = mapped_column(String(1024), nullable=False, comment="相对仓库路径")
    language: Mapped[str] = mapped_column(String(32), nullable=False, default="java")
    node_type: Mapped[str] = mapped_column(String(64), nullable=False, default="", comment="Tree-sitter 节点类型")
    symbol: Mapped[str] = mapped_column(String(512), nullable=False, default="", comment="类名/方法名")
    start_line: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", comment="text md5")
    char_len: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
