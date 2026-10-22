---
name: biz-sharding-sql
description: >-
  赢在一起(PCR)业务库分表与 SQL 路由：胎号 t_tyre、K1 扫码 t_scan_supplier_record*、
  K2 扫码 t_scan_store_record*；主数据 K1 查 t_shop_info、K2 查 t_store_info。
  Use when the user asks about 胎号、K1、K2、扫码记录、入出库流水、分表、物理表名,
  or when writing/debugging direct MySQL SQL against these tables.
compatibility: TraceMind diagnose tool_loop + biz readonly SQL; PCR/wintogether-bff sharding
metadata:
  author: TraceMind
  version: "1.0"
  scope: pcr-wintogether
---

# biz-sharding-sql

推理用分表与主数据路由；**勿写入对用户的最终回复**（业务受众下不要暴露表名/字段名）。

范围：PCR / 赢在一起。直连 MySQL 必须查**物理分表**，禁止查逻辑表（会 0 行）。

## 主数据（先查谁）

- K1（`CKH…`/`IZX…`）→ `t_shop_info`，键 `shop_code`
- K2（`S…`/`SLS…`）→ `t_store_info`，键 `org_code`/`store_code`；`account_num` = 所属 K1
- 关联：`t_store_info.account_num = t_shop_info.shop_code`
- 口诀：查 K1 先 `t_shop_info`；查 K2 先 `t_store_info`（用 `account_num` 得 parent/K1）

## 分表（PCR）

算法：`index = abs(Java String.hashCode(分片键)) % 分表数`（与 `app/agents/tools/biz_sharding.py` 一致）

| 业务 | 物理表 | 分片键 | 分表数 |
|------|--------|--------|--------|
| K2 扫码批次 | `t_scan_store_record_{0..63}` | `parent_code`（K1） | 64 |
| K2 扫码胎号明细 | `t_scan_store_record_tyre_{0..63}` | `parent_code`（K1） | 64 |
| K1 扫码批次 | `t_scan_supplier_record_{0..63}` | `org_code`（K1） | 64 |
| K1 扫码胎号明细 | `t_scan_supplier_record_tyre_{0..63}` | `org_code`（K1） | 64 |
| 胎号主档 | `t_tyre_{0..31}` | `tyre_code` | 32 |
| 胎号变更 | `t_tyre_change_{0..31}` | `tyre_code` | 32 |
| 胎号历史 | `t_tyre_history_record_{0..7}` | `tyre_code` | 8 |

注意：K2 扫码表里 `org_code`=K2；K1 扫码表里 `org_code`=K1。`t_scan_store_record_info_*` 倾向弃用，明细用 tyre 表。

## 查询路由

### K2 扫码 / 门店入出库

1. 已知 K1 → `parent_code` → 优先工具 `query_scan_tyre_records(tyre_code, parent_code)`
2. 仅知 K2 → `run_readonly_sql` 查 `t_store_info` 取 `account_num`，再作 parent 查扫码
3. 仅知胎号 → `query_scan_tyre_records(tyre_code)`（工具扫全分表）
4. 校验 K1 → `t_shop_info` WHERE `shop_code = parent_code`

### K1 扫码

K1 编码作 `org_code` → 物理表 `t_scan_supplier_record_tyre_{i}`；主数据 `t_shop_info`

### 胎号主档

`run_readonly_sql` 查 `t_tyre_{hash%32}`（需按算法算表名）

## 约束

- 禁止：`FROM t_scan_store_record_tyre` / `t_scan_supplier_record_tyre` / `t_tyre`（逻辑表）
- 门店扫码胎号明细优先 `query_scan_tyre_records`，不要用通用 SQL 硬扫逻辑表
- 只用本轮用户给出的胎号/K1/K2，禁止沿用上一轮实体

## Additional resources

- 字段释义、TBR 差异、权威配置路径：需要时用 `read_file` 读取 [references/reference.md](references/reference.md)
- TraceMind Deep Agent 通过 `skills=["/app/agents/skills/"]` 加载本 skill；激活后再读全文与 references
