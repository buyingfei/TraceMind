"""探测胎号/扫码相关表与 isAuto 字段。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from agents.tools.readonly_sql import get_biz_readonly_engine, run_select

get_settings.cache_clear()
get_biz_readonly_engine.cache_clear()


def main() -> None:
    tables = run_select(
        "SELECT TABLE_NAME FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE() AND ("
        "TABLE_NAME LIKE '%tyre%' OR TABLE_NAME LIKE '%tire%' OR "
        "TABLE_NAME LIKE '%scan%' OR TABLE_NAME LIKE '%barcode%' OR "
        "TABLE_NAME LIKE '%store_tyre%' OR TABLE_NAME LIKE '%stock%'"
        ") ORDER BY TABLE_NAME LIMIT 50"
    )
    print("=== tables ===")
    for row in tables.get("rows") or []:
        print(row["TABLE_NAME"])

    cols = run_select(
        "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND ("
        "COLUMN_NAME LIKE '%is_auto%' OR COLUMN_NAME LIKE '%isAuto%' OR "
        "COLUMN_NAME LIKE '%tyre_code%' OR COLUMN_NAME LIKE '%tyreCode%'"
        ") ORDER BY TABLE_NAME, COLUMN_NAME LIMIT 80"
    )
    print("=== interesting columns ===")
    for row in cols.get("rows") or []:
        print(row["TABLE_NAME"], row["COLUMN_NAME"])


if __name__ == "__main__":
    main()
