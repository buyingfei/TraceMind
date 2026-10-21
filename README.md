# diagnose-agent

业务问题排查 Agent：接收业务反馈 → 检索代码 → 只读库核对 → 输出可读结论。

## 目录结构

```text
diagnose-agent/
├── app/                     # Python 后端
│   ├── api/v1/
│   ├── agents/
│   ├── common/
│   ├── core/                # 配置、日志、prompts、code_index.yaml
│   ├── db/
│   ├── infra/
│   ├── schemas/
│   ├── services/
│   ├── main.py
│   ├── requirements.txt     # Python 依赖
│   ├── pyproject.toml       # 打包 / pytest 配置
│   └── .env.example
│   └── indexer/             # 代码入库批任务（非请求路径）
├── frontend/                # Vue 前端
├── scripts/
├── tests/
└── docker-compose.yml
```

分层约定：

- **infra**：创建连接/客户端（单例）
- **db/repositories**：本系统 MySQL CRUD
- **agents/tools**：可依赖 infra（业务只读库），不依赖 api
- **services**：编排 infra + repositories + agents
- **schemas / api**：边界 I/O

## 环境

推荐使用已有 conda 环境：

```bash
conda activate langgraph
cd D:\Codes\python\diagnose-agent
pip install -r app/requirements.txt
# 可选: cp app/.env.example app/.env  （也兼容仓库根目录 .env）
```

## 启动

后端：

```bash
python -m app.main
# 或: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

前端：

```bash
cd frontend
npm install
npm run dev
```

打开：

- 前端：http://localhost:5173
- API 文档：http://127.0.0.1:8000/docs
- Hello：http://127.0.0.1:8000/api/v1/hello?name=World

## 测试

```bash
pytest -c app/pyproject.toml -q
```

## Milvus（向量库）

仅支持 **Docker Milvus**（不使用本地 Lite）。

**重要：** 环境变量用 `DIAGNOSE_MILVUS_URI`，不要用 `MILVUS_URI`（会被 pymilvus 误读）。

```env
DIAGNOSE_MILVUS_URI=http://127.0.0.1:19530
```

需保证 `milvus-standalone` + etcd + minio 均为 healthy。联调：

```bash
python -m app.indexer.index_codebase --demo
# GET  /api/v1/milvus/health
# POST /api/v1/milvus/search  {"query":"连锁自动入库","top_k":3}
```
