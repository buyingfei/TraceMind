"""Export MySQL schema + chat table data into app/sql/."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text

from app.infra.db import get_engine

OUT = Path(__file__).resolve().parents[1] / "app" / "sql"
ALL_TABLES = [
    "chat_conversation",
    "chat_message",
    "chat_message_asset",
    "code_repo",
    "code_index_version",
    "code_chunk_meta",
]
CHAT_DATA_TABLES = [
    "chat_conversation",
    "chat_message",
    "chat_message_asset",
]


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, datetime):
        return "'" + value.strftime("%Y-%m-%d %H:%M:%S") + "'"
    if isinstance(value, date):
        return "'" + value.isoformat() + "'"
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    text_v = str(value)
    text_v = text_v.replace("\\", "\\\\").replace("'", "\\'")
    text_v = text_v.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return "'" + text_v + "'"


def dump_schema(conn) -> str:
    parts = [
        "-- TraceMind 本系统 MySQL schema（由当前库 SHOW CREATE TABLE 导出）",
        "-- 库名以 .env 中 MYSQL_DATABASE 为准（默认 diagnose-agent）",
        "SET NAMES utf8mb4;",
        "SET FOREIGN_KEY_CHECKS = 0;",
        "",
    ]
    for table in ALL_TABLES:
        row = conn.execute(text(f"SHOW CREATE TABLE `{table}`")).mappings().one()
        ddl = row.get("Create Table") or list(row.values())[1]
        parts.append("-- ----------------------------")
        parts.append(f"-- Table structure for `{table}`")
        parts.append("-- ----------------------------")
        parts.append(f"DROP TABLE IF EXISTS `{table}`;")
        parts.append(ddl + ";")
        parts.append("")
    parts.append("SET FOREIGN_KEY_CHECKS = 1;")
    parts.append("")
    return "\n".join(parts)


def dump_table_data(conn, table: str) -> str:
    cols = conn.execute(text(f"SHOW COLUMNS FROM `{table}`")).mappings().all()
    col_names = [c["Field"] for c in cols]
    rows = conn.execute(text(f"SELECT * FROM `{table}` ORDER BY id")).mappings().all()
    parts = [
        "-- ----------------------------",
        f"-- Records of `{table}` ({len(rows)} rows)",
        "-- ----------------------------",
    ]
    if not rows:
        parts.append("-- (empty)")
        parts.append("")
        return "\n".join(parts)

    col_list = ", ".join(f"`{c}`" for c in col_names)
    for row in rows:
        values = ", ".join(sql_literal(row[c]) for c in col_names)
        parts.append(f"INSERT INTO `{table}` ({col_list}) VALUES ({values});")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eng = get_engine()
    with eng.connect() as conn:
        (OUT / "01_schema.sql").write_text(dump_schema(conn), encoding="utf-8")

        data_parts = [
            "-- TraceMind 聊天相关表当前数据导出",
            "-- 表：chat_conversation / chat_message / chat_message_asset",
            "SET NAMES utf8mb4;",
            "SET FOREIGN_KEY_CHECKS = 0;",
            "",
        ]
        for table in CHAT_DATA_TABLES:
            data_parts.append(dump_table_data(conn, table))
        data_parts.append("SET FOREIGN_KEY_CHECKS = 1;")
        data_parts.append("")
        (OUT / "02_chat_data.sql").write_text("\n".join(data_parts), encoding="utf-8")

        meta_parts = [
            "-- TraceMind 代码索引元数据（小表全量；不含 code_chunk_meta 行数据）",
            "SET NAMES utf8mb4;",
            "SET FOREIGN_KEY_CHECKS = 0;",
            "",
        ]
        for table in ("code_repo", "code_index_version"):
            meta_parts.append(dump_table_data(conn, table))
        meta_parts.append("SET FOREIGN_KEY_CHECKS = 1;")
        meta_parts.append("")
        (OUT / "03_code_index_meta_data.sql").write_text(
            "\n".join(meta_parts), encoding="utf-8"
        )

    print("wrote", OUT)
    for p in sorted(OUT.glob("*.sql")):
        print(" ", p.name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
