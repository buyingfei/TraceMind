# TraceMind（diagnose-agent）

业务问题排查 Agent：接收业务反馈 → 检索代码 → 只读库核对 → 输出可读结论。

## 技术栈

### 后端 / API

| 类别 | 技术 | 说明 |
|------|------|------|
| Web 框架 | **FastAPI** + Uvicorn | HTTP API、OpenAPI 文档 |
| 流式输出 | SSE（`sse-starlette`） | 对话阶段推送（检索中 / 查库中等） |
| 配置校验 | Pydantic / pydantic-settings | `.env` 与 Settings |
| 包管理 | `app/requirements.txt`、`app/pyproject.toml` | 依赖与 pytest |

### ORM 与数据访问

| 类别 | 技术 | 说明 |
|------|------|------|
| ORM | **SQLAlchemy 2.0**（Declarative + `Mapped`） | 本系统表模型、`Session` CRUD |
| MySQL 驱动 | **PyMySQL** | `mysql+pymysql://...` |
| 本系统库 | MySQL | 会话、代码索引元数据等（`app/db`、`app/infra/db`） |
| 业务只读库 | MySQL（独立连接串） | Agent 查库核对；**仅 SELECT**（`app/infra/biz_db`） |
| 业务 SQL | SQLAlchemy `text()` | 扫码分表 / 质保 / 通用只读查询，非业务表 ORM 映射 |

本系统模型示例：`code_repo`、`code_index_version`、`code_chunk_meta`、对话相关表等（见 `app/db/models.py`）。

### 数据存储一览

| 存储 | 用途 | 客户端 / 说明 |
|------|------|----------------|
| **MySQL**（本系统） | 聊天会话、索引元数据、切块对账 | SQLAlchemy + PyMySQL |
| **MySQL**（业务只读） | 赢在一起扫码/门店/质保等核对 | 独立 Engine，超时/重试见配置 |
| **PostgreSQL** | LangGraph Agent 短期记忆 Checkpoint | `psycopg` + `langgraph-checkpoint-postgres` |
| **Milvus**（Docker） | 代码向量检索 | `pymilvus` / `langchain-milvus` |
| 本地文件 | 对话上传图等 | `data/chat_uploads` 等 |

> 向量在 Milvus；切块路径/版本等元数据在 MySQL，便于增量与对账。

### Agent / AI

| 类别 | 技术 | 说明 |
|------|------|------|
| 编排 | **LangChain**、**LangGraph** | 多轮状态、工具循环 |
| Deep Agent | **deepagents**（`create_deep_agent`） | Agent Skills 渐进加载 |
| 工具 | LangChain `@tool` | 扫码分表 / 质保 / 只读 SQL |
| 对话模型 | OpenAI 兼容网关（公司 gpt-sol） | 可扩展千问等 |
| Embedding | Ollama / OpenAI 兼容等 | 由 `MILVUS_EMBEDDING_PROVIDER` 配置 |
| OCR | **PaddleOCR**（PaddlePaddle） | 带图对话识字 |
| 代码切块 | Tree-sitter（Java） | 入库索引 |

### 前端

| 类别 | 技术 |
|------|------|
| 框架 | Vue 3 + Vite |
| 渲染 | marked + DOMPurify（Markdown） |

### 基础设施（可选）

- Docker Compose：可起 Postgres（profile `full`）；Milvus 建议官方 standalone + etcd + minio
- 环境变量：见 `app/.env.example`（也兼容仓库根目录 `.env`）

## 目录结构

```text
TraceMind/
├── app/                     # Python 后端
│   ├── api/v1/
│   ├── agents/              # diagnose 编排、tools、skills
│   ├── common/
│   ├── core/                # 配置、日志、code_index.yaml
│   ├── db/                  # SQLAlchemy 模型与 repositories
│   ├── infra/               # MySQL / 业务库 / LLM / OCR 客户端
│   ├── schemas/
│   ├── services/
│   ├── indexer/             # 代码入库批任务（非请求路径）
│   ├── sql/                 # 本系统 MySQL DDL + 聊天等数据导出
│   ├── main.py
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .env.example
├── frontend/                # Vue 前端
├── scripts/
├── tests/
└── docker-compose.yml
```

分层约定：

- **infra**：外部连接/客户端（单例）
- **db/repositories**：本系统 MySQL CRUD（SQLAlchemy ORM）
- **agents/tools**：可依赖 infra（业务只读库），不依赖 api
- **services**：编排 infra + repositories + agents
- **schemas / api**：边界 I/O

## 环境

推荐使用已有 conda 环境：

```bash
conda activate langgraph
cd D:\Codes\python\TraceMind
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

## 相关配置（数据与 Agent）

```env
# 本系统 MySQL（SQLAlchemy）
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=diagnose-agent

# 业务只读 MySQL（Agent 查库）
BIZ_READONLY_DATABASE_URL=mysql+pymysql://user:pass@host:3306/db?charset=utf8mb4
BIZ_READONLY_TIMEOUT_SEC=300
BIZ_READONLY_SQL_RETRIES=3

# LangGraph Checkpoint（PostgreSQL）
LANGGRAPH_DATABASE_URL=postgresql://postgres:123456@127.0.0.1:5432/langgraph_mem

# Agent 运行时
USE_DEEP_AGENT=true
```
