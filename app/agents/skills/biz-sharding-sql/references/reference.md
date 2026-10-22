# 分表规则补充参考

默认不注入模型上下文；需要字段释义或 TBR 差异时再读取。

## 字段语义

| 字段 | 出现位置 | 含义 |
|------|----------|------|
| `parent_code` | K2 扫码表族 | 所属 K1 |
| `org_code` | K2 扫码表族 | K2 门店 |
| `org_code` | K1 扫码表族 | K1（分片键） |
| `account_num` | `t_store_info` | 所属 K1 |
| `shop_code` | `t_shop_info` | K1 |
| `tyre_code` | 胎号表 | 胎号 |
| `is_auto` | 扫码胎号明细 | 1=系统/连锁自动 |

## TBR 差异

PCR K2 扫码按 `parent_code`(K1) 分 64 表；TBR 扫码按 `org_code`(K2) 分片，勿混用。

## 权威配置

`D:\Codes\wintogether-bff\wintogether-web-store\src\main\resources\application.properties`
