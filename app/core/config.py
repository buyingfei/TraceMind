"""应用配置。"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    # app/core/config.py → 项目根
    return Path(__file__).resolve().parents[2]


def _app_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_env_files() -> tuple[str, ...]:
    """优先 app/.env，其次仓库根 .env（兼容旧布局）。"""
    files: list[str] = []
    for path in (_app_dir() / ".env", _project_root() / ".env"):
        files.append(str(path))
    return tuple(files)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_default_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "diagnose-agent"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # MySQL（对应 Spring spring.datasource.*），库名 diagnose-agent
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="root")
    mysql_password: str = Field(default="123456")
    mysql_database: str = Field(default="diagnose-agent")
    mysql_charset: str = Field(default="utf8mb4")

    # OpenAI 兼容网关（对应 Spring AI spring.ai.openai.*）
    openai_base_url: str = Field(
        default="http://newapi.gtech.asia:3001/v1",
        description="需包含 /v1；若只配到端口，代码会自动补全",
    )
    openai_api_key: str = Field(default="", description="网关 API Key")
    openai_model: str = Field(default="gpt-5.6-sol")
    openai_temperature: float = Field(default=0.7)
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI 兼容网关上的 Embedding 模型名",
    )

    # 聊天仅用公司模型；失败按次数重试，不切换千问
    default_chat_model: str = Field(
        default="gpt-sol",
        description="默认聊天模型 key（仅 gpt-sol）",
    )
    llm_max_attempts: int = Field(
        default=3,
        description="公司模型调用最大尝试次数（含首次）",
    )

    # Milvus Docker（注意：不要用环境变量名 MILVUS_URI，会与 pymilvus 内置配置冲突）
    diagnose_milvus_uri: str = Field(
        default="http://127.0.0.1:19530",
        description="Docker Milvus 地址，例如 http://127.0.0.1:19530",
    )
    diagnose_milvus_token: str = Field(default="", description="本地 standalone 一般为空")
    diagnose_milvus_collection: str = Field(default="wt_code_chunks")
    milvus_embedding_dim: int = Field(default=1024)
    milvus_embedding_provider: str = Field(
        default="ollama",
        description="ollama | openai | local_hash",
    )

    # Ollama（本地 Embedding + 聊天）
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_embedding_model: str = Field(default="batiai/qwen3-embedding:0.6b")
    ollama_chat_model_qwen: str = Field(
        default="qwen2.5:7b-instruct-q4_K_M",
        description="本地 Qwen2.5 7B q4_K_M",
    )

    # 对话图片本地目录（相对项目根）
    chat_upload_dir: str = Field(default="data/chat_uploads")

    # 代码索引 / Tree-sitter 切块策略（YAML；本机仓库路径等放文件内）
    code_index_config_path: str = Field(
        default="app/core/code_index.yaml",
        description="相对项目根或绝对路径；可用环境变量 CODE_INDEX_CONFIG_PATH 覆盖",
    )

    # 启动时预热 PaddleOCR（加载模型 + 一次小图推理）
    ocr_warmup_on_startup: bool = Field(
        default=True,
        description="应用启动时预热 OCR，避免首条带图消息冷启动过慢",
    )

    # LangGraph Checkpointer（Postgres，与 MySQL 聊天库分离）
    langgraph_database_url: str = Field(
        default="postgresql://postgres:123456@127.0.0.1:5432/langgraph_mem",
        description="Agent 短期记忆 checkpoint；psycopg 连接串",
    )
    agent_memory_max_messages: int = Field(
        default=20,
        description="喂给模型的最近消息条数上限（含 user/assistant）",
    )

    # 回答受众：business=业务话术不露代码；technical=可含 path/行号
    answer_audience: str = Field(
        default="business",
        description="business | technical",
    )

    # 业务只读库（模型核对数据用；勿与本系统 MySQL 混用）
    biz_readonly_database_url: str = Field(
        default="",
        description="SQLAlchemy URL，如 mysql+pymysql://user:pass@host:3306/db?charset=utf8mb4",
    )
    biz_readonly_sql_enabled: bool = Field(
        default=True,
        description="是否允许模型调用只读 SQL 工具",
    )
    biz_readonly_max_rows: int = Field(default=30, description="单次查询最大返回行数")
    biz_readonly_timeout_sec: int = Field(default=15, description="单次查询超时秒数")
    agent_max_tool_rounds: int = Field(
        default=3,
        description="单轮对话最多工具调用轮数（查库等）",
    )

    @property
    def milvus_uri(self) -> str:
        return self.diagnose_milvus_uri

    @property
    def milvus_token(self) -> str:
        return self.diagnose_milvus_token

    @property
    def milvus_collection(self) -> str:
        return self.diagnose_milvus_collection

    def normalized_openai_base_url(self) -> str:
        """Spring 的 base-url 通常不含 /v1，Python SDK 一般需要带 /v1。"""
        url = self.openai_base_url.rstrip("/")
        if url.endswith("/v1"):
            return url
        return f"{url}/v1"

    @property
    def database_url(self) -> str:
        """SQLAlchemy MySQL URL（PyMySQL）。"""
        user = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password)
        database = quote_plus(self.mysql_database)
        return (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.mysql_host}:{self.mysql_port}/{database}"
            f"?charset={self.mysql_charset}"
        )

    def resolve_code_index_config_path(self) -> Path:
        path = Path(self.code_index_config_path)
        if not path.is_absolute():
            path = _project_root() / path
        return path.resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
