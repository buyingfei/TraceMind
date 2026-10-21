"""烟测：按 parent / 仅胎号查询扫码分表。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from agents.tools.biz_sharding import store_scan_shard_index, scan_store_tyre_table
from agents.tools.readonly_sql import get_biz_readonly_engine
from agents.tools.scan_tyre_query import query_scan_tyre_records

get_settings.cache_clear()
get_biz_readonly_engine.cache_clear()

TYRE = "9001462521"
PARENT = "CKH0024005"


def main() -> None:
    print("shard", store_scan_shard_index(PARENT), scan_store_tyre_table(PARENT))
    r1 = query_scan_tyre_records(TYRE, parent_code=PARENT)
    print("by_parent", r1["row_count"], r1["route"])
    for row in r1["rows"]:
        print(
            row.get("CREATE_TIME"),
            row.get("ORG_CODE"),
            row.get("RECORD_TYPE"),
            row.get("is_auto"),
            row.get("shard_table"),
        )
    r2 = query_scan_tyre_records(TYRE)
    print("scan_all", r2["row_count"], r2["route"].get("mode"), r2["route"].get("shard_table"))


if __name__ == "__main__":
    main()
