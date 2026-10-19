"""应用配置。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "diagnose-agent"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

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

    # Milvus Docker（注意：不要用环境变量名 MILVUS_URI，会与 pymilvus 内置配置冲突）
    diagnose_milvus_uri: str = Field(
        default="http://127.0.0.1:19530",
        description="Docker Milvus 地址，例如 http://127.0.0.1:19530",
    )
    diagnose_milvus_token: str = Field(default="", description="本地 standalone 一般为空")
    diagnose_milvus_collection: str = Field(default="diagnose_code_chunks")
    milvus_embedding_dim: int = Field(default=1536)
    milvus_embedding_provider: str = Field(
        default="openai",
        description="openai | local_hash；local_hash 仅用于连通性测试",
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
