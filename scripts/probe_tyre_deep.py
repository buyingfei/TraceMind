"""深挖胎号落库位置与 is_auto 真实情况。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.core.config import get_settings
from agents.tools.readonly_sql import get_biz_readonly_engine, run_select

get_settings.cache_clear()
get_biz_readonly_engine.cache_clear()

TYRE = "9001462521"


def main() -> None:
    engine = get_biz_readonly_engine()
    with engine.connect() as conn:
        db = conn.execute(text("SELECT DATABASE() AS d")).mappings().one()["d"]
        print("database", db)

        # is_auto 元数据
        meta = conn.execute(
            text(
                "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                "FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME LIKE 't_scan_store_record_tyre%' "
                "AND COLUMN_NAME LIKE '%auto%' "
                "ORDER BY TABLE_NAME, COLUMN_NAME"
            )
        ).mappings().all()
        print("auto_columns", len(meta))
        for r in meta[:10]:
            print(dict(r))

        # 全部分表
        tables = conn.execute(
            text(
                "SELECT TABLE_NAME FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME LIKE 't_scan_store_record_tyre%' "
                "ORDER BY TABLE_NAME"
            )
        ).scalars().all()
        print("all_tyre_tables", len(tables))

        found = 0
        for name in tables:
            # 先确认列
            cols = {
                c.upper()
                for c in conn.execute(
                    text(
                        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
                    ),
                    {"t": name},
                ).scalars()
            }
            has_auto = "IS_AUTO" in cols
            select_cols = "ID, TYRE_CODE, ORG_CODE, SCAN_TYPE, RECORD_TYPE, CREATE_TIME"
            if has_auto:
                select_cols += ", is_auto"
            sql = (
                f"SELECT {select_cols} FROM `{name}` "
                f"WHERE TYRE_CODE = :tyre LIMIT 20"
            )
            rows = conn.execute(text(sql), {"tyre": TYRE}).mappings().all()
            if rows:
                found += len(rows)
                print("HIT", name, "auto=", has_auto, "n=", len(rows))
                for r in rows:
                    print(" ", dict(r))

        print("total_hits", found)

        # 其它可能表
        candidates = [
            "t_commerce_tyre",
            "t_scan_store_record_tyre",
            "k1_tyre_code",
            "t_home_data_scan",
        ]
        for name in candidates:
            exists = conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
                ),
                {"t": name},
            ).scalar()
            if not exists:
                print("skip missing", name)
                continue
            cols = [
                c
                for c in conn.execute(
                    text(
                        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
                    ),
                    {"t": name},
                ).scalars()
            ]
            tyre_cols = [c for c in cols if "tyre" in c.lower() or "TYRE" in c]
            print("table", name, "tyre_cols", tyre_cols)
            for tc in tyre_cols[:2]:
                try:
                    rows = conn.execute(
                        text(f"SELECT * FROM `{name}` WHERE `{tc}` = :tyre LIMIT 5"),
                        {"tyre": TYRE},
                    ).mappings().all()
                    print(" ", tc, "rows", len(rows))
                    for r in rows[:3]:
                        print("  ", {k: r[k] for k in list(r.keys())[:12]})
                except Exception as exc:
                    print(" ", tc, "err", exc)

        # 任意分表近期有无数据（确认只读库是否空）
        sample = "t_scan_store_record_tyre_0"
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM `{sample}`")).scalar()
        recent = conn.execute(
            text(
                f"SELECT TYRE_CODE, ORG_CODE, CREATE_TIME FROM `{sample}` "
                f"ORDER BY CREATE_TIME DESC LIMIT 3"
            )
        ).mappings().all()
        print("sample_count", sample, cnt)
        for r in recent:
            print(" recent", dict(r))

        # 2026-04-29 前后有没有数据
        day_cnt = conn.execute(
            text(
                f"SELECT COUNT(*) FROM `{sample}` "
                f"WHERE CREATE_TIME >= '2026-04-29' AND CREATE_TIME < '2026-04-30'"
            )
        ).scalar()
        print("sample_day_2026-04-29", day_cnt)


if __name__ == "__main__":
    main()
