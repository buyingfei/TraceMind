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

    def normalized_openai_base_url(self) -> str:
        """Spring 的 base-url 通常不含 /v1，Python SDK 一般需要带 /v1。"""
        url = self.openai_base_url.rstrip("/")
        if url.endswith("/v1"):
            return url
        return f"{url}/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
