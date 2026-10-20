"""Milvus Collection 模型定义（pymilvus MilvusClient）。

对应设计：wt_code_chunks —— 代码切块向量，按 repo/branch/image_tag/index_version 隔离。
"""

from __future__ import annotations

from typing import Any

from pymilvus import DataType, MilvusClient

# Collection 名（与 code_index_version.milvus_collection / .env 对齐）
WT_CODE_CHUNKS = "wt_code_chunks"

# VARCHAR 上限
MAX_ID = 64
MAX_REPO = 64
MAX_BRANCH = 128
MAX_IMAGE_TAG = 128
MAX_INDEX_VERSION = 64
MAX_GIT_COMMIT = 40
MAX_PATH = 1024
MAX_MODULE = 128
MAX_LANGUAGE = 32
MAX_NODE_TYPE = 64
MAX_SYMBOL = 512
MAX_TEXT = 8192
MAX_DOC_TYPE = 32
MAX_CONTENT_HASH = 64

# 检索时默认回传字段（不含 embedding）
OUTPUT_FIELDS = [
    "repo",
    "branch",
    "image_tag",
    "index_version",
    "git_commit",
    "path",
    "module",
    "language",
    "node_type",
    "symbol",
    "start_line",
    "end_line",
    "text",
    "doc_type",
    "content_hash",
]


def build_code_chunks_schema(embedding_dim: int):
    """构建代码切块 Collection Schema。"""
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)

    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=MAX_ID,
        description="md5(repo|image_tag|index_version|path|start_line|end_line|node_type)",
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=embedding_dim,
        description="代码块向量",
    )

    # 版本维度（检索必过滤）
    schema.add_field(field_name="repo", datatype=DataType.VARCHAR, max_length=MAX_REPO)
    schema.add_field(field_name="branch", datatype=DataType.VARCHAR, max_length=MAX_BRANCH)
    schema.add_field(field_name="image_tag", datatype=DataType.VARCHAR, max_length=MAX_IMAGE_TAG)
    schema.add_field(field_name="index_version", datatype=DataType.VARCHAR, max_length=MAX_INDEX_VERSION)
    schema.add_field(field_name="git_commit", datatype=DataType.VARCHAR, max_length=MAX_GIT_COMMIT)

    # 代码定位
    schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=MAX_PATH)
    schema.add_field(field_name="module", datatype=DataType.VARCHAR, max_length=MAX_MODULE)
    schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=MAX_LANGUAGE)
    schema.add_field(field_name="node_type", datatype=DataType.VARCHAR, max_length=MAX_NODE_TYPE)
    schema.add_field(field_name="symbol", datatype=DataType.VARCHAR, max_length=MAX_SYMBOL)
    schema.add_field(field_name="start_line", datatype=DataType.INT64)
    schema.add_field(field_name="end_line", datatype=DataType.INT64)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=MAX_TEXT)

    # 扩展
    schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=MAX_DOC_TYPE)
    schema.add_field(field_name="content_hash", datatype=DataType.VARCHAR, max_length=MAX_CONTENT_HASH)

    return schema


def build_code_chunks_index_params(client: MilvusClient):
    """向量索引 + 常用过滤字段标量索引。"""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    # 标量索引便于按镜像/批次过滤（Milvus 2.4+ 支持）
    for field in ("repo", "branch", "image_tag", "index_version", "doc_type"):
        index_params.add_index(field_name=field, index_type="INVERTED")
    return index_params


def ensure_code_chunks_collection(
    client: MilvusClient,
    *,
    collection_name: str = WT_CODE_CHUNKS,
    embedding_dim: int = 1536,
    drop_if_exists: bool = False,
) -> str:
    """按模型创建 collection（已存在则跳过）。"""
    if drop_if_exists and client.has_collection(collection_name):
        client.drop_collection(collection_name)

    if client.has_collection(collection_name):
        return collection_name

    schema = build_code_chunks_schema(embedding_dim)
    index_params = build_code_chunks_index_params(client)
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    return collection_name


def empty_code_chunk_row() -> dict[str, Any]:
    """空行模板，写入前填充。"""
    return {
        "id": "",
        "embedding": [],
        "repo": "",
        "branch": "",
        "image_tag": "",
        "index_version": "",
        "git_commit": "",
        "path": "",
        "module": "",
        "language": "java",
        "node_type": "",
        "symbol": "",
        "start_line": 0,
        "end_line": 0,
        "text": "",
        "doc_type": "code",
        "content_hash": "",
    }
