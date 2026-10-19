# diagnose-agent

业务问题排查 Agent：接收业务反馈 → 检索代码 → 只读库核对 → 输出可读结论。

当前为**项目骨架 + Hello World**，尚未接入大模型 / Milvus / 业务库。

## 目录结构

```text
diagnose-agent/
├── app/                     # FastAPI 应用
│   ├── api/v1/              # REST：hello / 工单占位
│   ├── core/                # 配置、日志、异常、prompts
│   ├── services/            # embedding / rerank / 切块 / OSS（占位）
│   └── db/                  # 本系统库 Repository（占位）
├── agents/
│   ├── diagnose/            # 主 Graph（骨架：Hello StateGraph）
│   ├── tools/               # 搜代码 / 读文件 / 只读 SQL（占位）
│   └── playbooks/           # 场景剧本
├── indexer/                 # 代码入库批任务（占位）
├── web/                     # Vue 前端（后续）
├── tests/
├── docker-compose.yml       # Postgres + Milvus
├── requirements.txt         # 骨架最小依赖
└── requirements_full.txt    # 完整依赖（后续）
```

## 环境

推荐使用已有 conda 环境：

```bash
conda activate langgraph
cd D:\Codes\python\diagnose-agent
pip install -r requirements.txt
```

## 启动

```bash
python main.py
```

打开：

- 首页：http://127.0.0.1:8000/
- 健康检查：http://127.0.0.1:8000/health
- Hello：http://127.0.0.1:8000/api/v1/hello?name=World
- Swagger：http://127.0.0.1:8000/docs

## 测试

```bash
pytest -q
```

## 下一步（未做）

1. 接入 DashScope（Qwen）与 Prompt
2. 实现 diagnose 正式流水线节点
3. 代码入库 Milvus + ripgrep 工具
4. 业务库只读 SQL
5. Vue 工单页
