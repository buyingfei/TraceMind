from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.core.config import get_settings
from agents.tools.readonly_sql import get_biz_readonly_engine

get_settings.cache_clear()
get_biz_readonly_engine.cache_clear()

TYRE = "9001462521"


def main() -> None:
    eng = get_biz_readonly_engine()
    with eng.connect() as c:
        print("database", c.execute(text("SELECT DATABASE()")).scalar())
        n0 = c.execute(text("SELECT COUNT(*) FROM t_scan_store_record_tyre_0")).scalar()
        print("shard0_total", n0)
        sample = c.execute(
            text(
                "SELECT TYRE_CODE, ORG_CODE, CREATE_TIME, is_auto "
                "FROM t_scan_store_record_tyre_0 ORDER BY CREATE_TIME DESC LIMIT 2"
            )
        ).mappings().all()
        print("shard0_sample", [dict(x) for x in sample])

        hit = 0
        for i in range(64):
            name = f"t_scan_store_record_tyre_{i}"
            try:
                n = c.execute(
                    text(f"SELECT COUNT(*) FROM `{name}` WHERE TYRE_CODE = :v"),
                    {"v": TYRE},
                ).scalar()
            except Exception:
                continue
            if n:
                print("HIT", name, n)
                hit += 1
        print("hit_shards", hit)

        # 逻辑表
        n = c.execute(
            text("SELECT COUNT(*) FROM t_scan_store_record_tyre WHERE TYRE_CODE = :v"),
            {"v": TYRE},
        ).scalar()
        print("logical_table_count", n)


if __name__ == "__main__":
    main()
