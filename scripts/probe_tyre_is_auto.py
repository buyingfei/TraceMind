"""查胎号 9001462521 扫码明细 is_auto。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from agents.tools.readonly_sql import get_biz_readonly_engine, run_select

get_settings.cache_clear()
get_biz_readonly_engine.cache_clear()

TYRE = "9001462521"


def main() -> None:
    # 找出有 is_auto 的分表
    shards = run_select(
        "SELECT TABLE_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND COLUMN_NAME = 'is_auto' "
        "AND TABLE_NAME LIKE 't_scan_store_record_tyre%' "
        "ORDER BY TABLE_NAME LIMIT 40"
    )
    names = [r["TABLE_NAME"] for r in (shards.get("rows") or [])]
    print("shards_with_is_auto", len(names))

    # 列结构抽样
    sample = names[0] if names else "t_scan_store_record_tyre_0"
    cols = run_select(
        "SELECT COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS "
        f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{sample}' "
        "ORDER BY ORDINAL_POSITION LIMIT 40"
    )
    print("=== columns", sample, "===")
    for c in cols.get("rows") or []:
        print(c["COLUMN_NAME"], c["DATA_TYPE"])

    # 分表 UNION 查胎号（限制分表数量）
    parts = []
    for name in names[:32]:
        parts.append(
            "SELECT '"
            + name
            + "' AS shard, TYRE_CODE, org_code, parent_code, "
            "operate_type, is_auto, create_time, id "
            f"FROM `{name}` WHERE TYRE_CODE = '{TYRE}'"
        )
    if not parts:
        print("no shards")
        return
    sql = " UNION ALL ".join(parts) + " ORDER BY create_time LIMIT 30"
    # run_select wraps with subquery LIMIT - ORDER BY outside may break; use plain
    try:
        from agents.tools.readonly_sql import _normalize_sql, get_biz_readonly_engine
        from sqlalchemy import text

        engine = get_biz_readonly_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        print("=== rows", len(rows), "===")
        for r in rows:
            print(dict(r))
    except Exception as exc:
        print("query failed", exc)
        # fallback: try logical table
        try:
            r = run_select(
                f"SELECT TYRE_CODE, org_code, is_auto, operate_type, create_time "
                f"FROM t_scan_store_record_tyre WHERE TYRE_CODE = '{TYRE}' LIMIT 20"
            )
            print("logical", r)
        except Exception as exc2:
            print("logical failed", exc2)


if __name__ == "__main__":
    main()
