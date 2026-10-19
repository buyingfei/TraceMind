"""代码库索引入口（最小可用：写入演示片段，验证 Milvus）。

完整扫描仓库后续再做。当前支持:

    python -m indexer.index_codebase --demo
"""

from __future__ import annotations

import argparse
import hashlib

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services import milvus_service


def _demo_chunks() -> list[dict[str, str]]:
    samples = [
        {
            "path": "TyreStatusVerifierService.java",
            "repo": "wintogether-bff",
            "text": (
                "连锁店场景：当轮胎状态为 K2 入库且操作门店与 tyre.k2Org 不一致时，"
                "调用 isChainStore 判断连锁关系，随后 createOut 原门店自动出库，"
                "createIn 当前门店自动入库。方法名 k2InAndOrgCodeNotEquals。"
            ),
        },
        {
            "path": "RetailCommService.java",
            "repo": "wintogether-bff",
            "text": (
                "isChainStore 通过 StoreRelationShip 判断两店是否连锁且同一 K1；"
                "createIn/createOut 写入扫码记录并设置 isAuto=1。"
            ),
        },
        {
            "path": "hello.md",
            "repo": "diagnose-agent",
            "text": "diagnose-agent Hello World skeleton without LLM.",
        },
    ]
    chunks: list[dict[str, str]] = []
    for item in samples:
        raw_id = f"{item['repo']}:{item['path']}:{item['text'][:48]}"
        chunk_id = hashlib.md5(raw_id.encode("utf-8")).hexdigest()
        chunks.append(
            {
                "id": chunk_id,
                "text": item["text"],
                "path": item["path"],
                "repo": item["repo"],
            }
        )
    return chunks


def index_demo() -> int:
    settings = get_settings()
    milvus_service.ensure_collection(settings=settings)
    return milvus_service.upsert_chunks(_demo_chunks(), settings=settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="diagnose-agent indexer")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="写入内置演示片段，用于验证 Milvus 检索",
    )
    args = parser.parse_args()

    setup_logging(get_settings().log_level)
    if not args.demo:
        parser.error("当前仅支持 --demo；完整仓库索引稍后实现")

    count = index_demo()
    print(f"upserted={count}, collection={get_settings().milvus_collection}")


if __name__ == "__main__":
    main()
