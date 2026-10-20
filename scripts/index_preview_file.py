"""索引 preview Java 文件到 Milvus（Ollama Embedding）。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("MILVUS_URI", None)

from app.core.config import get_settings
from app.services.code_index_service import index_java_file
from app.services.milvus_service import get_milvus_client, search


def main() -> None:
    get_settings.cache_clear()
    get_milvus_client.cache_clear()

    result = index_java_file(recreate_collection=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    hits = search(
        "SpringBootApplication 启动类 main 方法",
        top_k=3,
        image_tag=result["version"]["image_tag"],
        index_version=result["version"]["index_version"],
        repo=result["version"]["repo"],
    )
    print("---- search ----")
    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
