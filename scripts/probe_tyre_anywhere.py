"""在只读库中广搜胎号 9001462521 落在哪些表。"""

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
    engine = get_biz_readonly_engine()
    with engine.connect() as conn:
        cols = conn.execute(
            text(
                "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND ("
                "COLUMN_NAME IN ('TYRE_CODE','tyre_code','TyreCode') "
                "OR COLUMN_NAME LIKE '%TYRE%CODE%' OR COLUMN_NAME LIKE '%tyre%code%'"
                ") ORDER BY TABLE_NAME"
            )
        ).fetchall()
        print("tyre-code columns", len(cols))
        hits = []
        for table, col in cols:
            try:
                n = conn.execute(
                    text(f"SELECT COUNT(*) FROM `{table}` WHERE `{col}` = :v"),
                    {"v": TYRE},
                ).scalar()
            except Exception as exc:
                print("err", table, col, exc)
                continue
            if n:
                print(f"HIT {table}.{col} count={n}")
                # sample
                rows = conn.execute(
                    text(f"SELECT * FROM `{table}` WHERE `{col}` = :v LIMIT 3"),
                    {"v": TYRE},
                ).mappings().all()
                for r in rows:
                    d = dict(r)
                    # 截断
                    slim = {k: d[k] for k in list(d)[:12]}
                    print(" ", slim)
                hits.append((table, col, n))
        print("total_hit_tables", len(hits))
        for h in hits:
            print(h)

        # DB name / host hint
        db = conn.execute(text("SELECT DATABASE()")).scalar()
        print("database", db)


if __name__ == "__main__":
    main()
