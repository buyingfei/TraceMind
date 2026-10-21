"""核对胎号 9001462521 在只读库中的真实落点。"""

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

    # 1) 含 TYRE_CODE / tyre_code / is_auto 的表
    meta = run_select(
        "SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND ("
        "COLUMN_NAME IN ('TYRE_CODE','tyre_code','is_auto','IS_AUTO') "
        "OR COLUMN_NAME LIKE '%tyre%code%'"
        ") ORDER BY TABLE_NAME, COLUMN_NAME LIMIT 120"
    )
    print("=== meta sample ===")
    for r in meta.get("rows") or []:
        print(r["TABLE_NAME"], r["COLUMN_NAME"])

    shards = run_select(
        "SELECT TABLE_NAME FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE 't_scan_store_record_tyre%' "
        "ORDER BY TABLE_NAME"
    )
    names = [r["TABLE_NAME"] for r in (shards.get("rows") or [])]
    print("shard_count", len(names))

    # 2) 看一表是否有 is_auto
    sample = "t_scan_store_record_tyre_0"
    cols = run_select(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        f"WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{sample}' "
        "AND COLUMN_NAME LIKE '%auto%'"
    )
    print("auto cols on _0", cols.get("rows"))

    # 3) 逐分表查胎号（用真实列）
    found = []
    with engine.connect() as conn:
        for name in names:
            # 探列
            col_rows = conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
                ),
                {"t": name},
            ).fetchall()
            colnames = {r[0] for r in col_rows}
            tyre_col = "TYRE_CODE" if "TYRE_CODE" in colnames else (
                "tyre_code" if "tyre_code" in colnames else None
            )
            if not tyre_col:
                continue
            auto_col = "is_auto" if "is_auto" in colnames else (
                "IS_AUTO" if "IS_AUTO" in colnames else None
            )
            select_cols = [tyre_col, "ORG_CODE", "CREATE_TIME", "SCAN_TYPE", "RECORD_TYPE"]
            select_cols = [c for c in select_cols if c in colnames]
            if auto_col:
                select_cols.append(auto_col)
            sql = (
                f"SELECT {', '.join('`'+c+'`' for c in select_cols)} "
                f"FROM `{name}` WHERE `{tyre_col}` = :tyre LIMIT 20"
            )
            rows = conn.execute(text(sql), {"tyre": TYRE}).mappings().all()
            if rows:
                print(f"FOUND in {name}: {len(rows)}")
                for row in rows:
                    print(dict(row))
                found.append(name)

    if not found:
        print("NOT FOUND in any t_scan_store_record_tyre*")

    # 4) 其他可能表
    for table, col in [
        ("t_scan_store_record", "TYRE_CODE"),
        ("t_commerce_tyre", "tyre_code"),
        ("k1_tyre_code", "TYRE_CODE"),
    ]:
        try:
            with engine.connect() as conn:
                exists = conn.execute(
                    text(
                        "SELECT COUNT(*) AS c FROM information_schema.TABLES "
                        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"
                    ),
                    {"t": table},
                ).scalar()
                if not exists:
                    print(f"skip missing {table}")
                    continue
                # 主表可能没有 tyre 列
                col_ok = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t "
                        "AND COLUMN_NAME = :c"
                    ),
                    {"t": table, "c": col},
                ).scalar()
                if not col_ok:
                    print(f"{table} no column {col}")
                    continue
                n = conn.execute(
                    text(f"SELECT COUNT(*) AS c FROM `{table}` WHERE `{col}` = :tyre"),
                    {"tyre": TYRE},
                ).scalar()
                print(f"{table}.{col} count={n}")
        except Exception as exc:
            print(f"{table} err: {exc}")

    # 5) 模拟模型常犯的错 SQL
    print("=== bad sql simulations ===")
    for sql in [
        f"SELECT * FROM t_scan_store_record_tyre WHERE TYRE_CODE = '{TYRE}' LIMIT 20",
        f"SELECT * FROM t_scan_store_record WHERE TYRE_CODE = '{TYRE}' LIMIT 20",
        f"SELECT TYRE_CODE, is_auto FROM t_scan_store_record_tyre_0 WHERE TYRE_CODE = '{TYRE}' LIMIT 20",
    ]:
        try:
            r = run_select(sql)
            print("OK", sql[:70], "rows", r.get("row_count"))
        except Exception as exc:
            print("FAIL", sql[:70], "->", exc)


if __name__ == "__main__":
    main()
