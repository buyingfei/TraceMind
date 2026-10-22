---
name: biz-sql-query
description: >-
  业务只读 SQL 查询规范：所有 SELECT/查库工具（run_readonly_sql、query_scan_tyre_records、
  query_warranty_by_plate）的超时、重试与安全约束。
  Use when running any business SQL, 查库, 只读查询, SELECT, timeout, or retry.
compatibility: TraceMind biz readonly SQL; timeout 5min; retries 3
metadata:
  author: TraceMind
  version: "1.0"
  scope: all-sql
---

# biz-sql-query

适用于 **所有** 业务只读 SQL（含通用 SQL、扫码分表、质保车牌）。推理用；勿在对用户最终回复中粘贴原始 SQL。

## 执行策略（运行时已落地）

| 项 | 值 | 说明 |
|----|----|------|
| 单次超时 | **5 分钟**（300s） | `MAX_EXECUTION_TIME` + 连接 read_timeout |
| 失败重试 | **3 次** | 不含首次，最多共执行 4 次 |
| 语句类型 | 仅 `SELECT` / `WITH … SELECT` | 禁止写操作 |

配置项：`BIZ_READONLY_TIMEOUT_SEC=300`、`BIZ_READONLY_SQL_RETRIES=3`。

## 选用哪个工具

1. 门店扫码胎号流水 → `query_scan_tyre_records`（自动分表）
2. 车牌质保卡 → `query_warranty_by_plate`
3. 其它主数据 / 物理分表 / 连锁关系 → `run_readonly_sql`
4. 分表路由细节 → 同时遵循 skill `biz-sharding-sql`

## 编写 SQL 约束

- 禁止逻辑表直查（无中间件时 0 行）：`t_scan_store_record_tyre`、`t_scan_supplier_record_tyre`、`t_tyre` 等
- 单条语句、不要分号；优先带分片键 / 主键条件
- 只用本轮用户给出的业务键，禁止沿用上一轮实体
- 超时或重试耗尽时：向用户说明未能用库核对，不要编造行数据

## 失败处理

- 临时网络/超时类错误：运行时自动重试，模型无需自行循环调工具凑重试
- 仍失败：报告错误摘要，可建议缩小条件或补 K1/`parent_code` 后再查
