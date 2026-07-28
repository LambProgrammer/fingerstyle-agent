"""全局配置：用 pydantic-settings 封装 .env。

全项目统一通过 `from src.config import settings` 取配置，
好处：启动即校验类型（fail-fast）、IDE 补全、配置来源单一。
密钥类字段默认空字符串，是否缺失由使用方在调用点校验并给出明确报错。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM: DeepSeek（决策层：Agent 3 / Agent 5）---
    deepseek_api_key: str = ""

    # --- LangSmith 可观测性（开发期默认关闭，里程碑 4 起开启）---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "fingerstyle-agent"

    # --- PostgreSQL（短期记忆 Checkpointer，里程碑 8 使用）---
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "fingerstyle"
    postgres_password: str = "fingerstyle_dev_pw"
    postgres_db: str = "fingerstyle"

    # --- Redis（长期记忆 Store，里程碑 8 使用）---
    redis_host: str = "redis"
    redis_port: int = 6379

    # --- Chroma（RAG 向量库，里程碑 5 使用）---
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    # --- 应用 ---
    log_level: str = "INFO"


settings = Settings()
