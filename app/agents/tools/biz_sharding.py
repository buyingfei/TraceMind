"""业务库分表索引（对齐 wintogether StoreScanPreciseShardingAlgorithm）。

分片键：parent_code
算法：Java String.hashCode() 取绝对值后对分表数取模。
"""

from __future__ import annotations

# t_scan_store_record_tyre_$->{0..63}
SCAN_STORE_TYRE_SHARD_COUNT = 64
SCAN_STORE_TYRE_TABLE_PREFIX = "t_scan_store_record_tyre"


def java_string_hash_code(value: str) -> int:
    """等价于 Java String.hashCode()（32 位有符号）。"""
    h = 0
    for ch in value:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return h


def store_scan_shard_index(parent_code: str, *, shard_count: int = SCAN_STORE_TYRE_SHARD_COUNT) -> int:
    """按 parent_code 计算扫码分表序号（0..shard_count-1）。"""
    code = (parent_code or "").strip()
    if not code:
        raise ValueError("parent_code 为空，无法计算分表")
    if shard_count <= 0:
        raise ValueError("shard_count 无效")
    h = java_string_hash_code(code)
    if h < 0:
        # 与 Java BasicPreciseShardingAlgorithm 一致；MIN_VALUE 取负仍溢出为自身
        h = -h
    return h % shard_count


def scan_store_tyre_table(parent_code: str) -> str:
    idx = store_scan_shard_index(parent_code)
    return f"{SCAN_STORE_TYRE_TABLE_PREFIX}_{idx}"


def all_scan_store_tyre_tables(*, shard_count: int = SCAN_STORE_TYRE_SHARD_COUNT) -> list[str]:
    return [f"{SCAN_STORE_TYRE_TABLE_PREFIX}_{i}" for i in range(shard_count)]
