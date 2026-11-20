"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed runtime configuration.

    Relative data paths are always resolved from the project root, so CLI, API,
    tests, and process managers behave consistently regardless of their cwd.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "LangChain Customer Service Agent"
    app_env: Literal["dev", "test", "prod"] = "dev"
    company_name: str = "你的公司名称"
    support_email: str = "support@example.com"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Chat and embedding providers are separated because they are often different services.
    chat_api_key: SecretStr | None = None
    chat_base_url: str | None = None
    chat_model_name: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    embedding_model_name: str | None = None

    # Backward-compatible OpenAI-style names. New deployments should prefer the fields above.
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model_name: str | None = None
    openai_embedding_model_name: str | None = None

    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    agent_verbose: bool = False
    # multi 为新版协作系统；single 保留原版，便于学习和效果对照。
    agent_mode: Literal["single", "multi"] = "multi"
    multi_agent_max_concurrency: int = Field(default=3, ge=1, le=12)
    multi_agent_task_timeout_seconds: float = Field(default=45.0, gt=0, le=300)
    multi_agent_max_attempts: int = Field(default=2, ge=1, le=3)
    supervisor_timeout_seconds: float = Field(default=45.0, gt=0, le=300)
    agent_recursion_limit: int = Field(default=20, ge=4, le=100)
    model_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    model_max_retries: int = Field(default=2, ge=0, le=10)
    embedding_check_ctx_length: bool = False

    memory_db_path: Path = Path("data/agent_checkpoints.sqlite3")
    # 两种图的状态结构不同，分别保存，避免旧 checkpoint 被新图误读。
    multi_agent_db_path: Path = Path("data/multi_agent_checkpoints.sqlite3")
    after_sales_db_path: Path = Path("data/after_sales.sqlite3")
    profile_db_path: Path = Path("data/customer_profiles.sqlite3")
    rag_db_path: Path = Path("data/rag_chunks.sqlite3")
    ticket_db_path: Path = Path("data/tickets.sqlite3")
    knowledge_base_dir: Path = Path("data/knowledge_base")
    sqlite_busy_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    memory_recent_turns: int = Field(default=6, ge=1, le=50)
    memory_summary_enabled: bool = True
    memory_summary_max_chars: int = Field(default=4000, ge=500, le=20_000)
    memory_summary_input_max_chars: int = Field(default=20_000, ge=2000, le=100_000)
    # 长期画像覆盖更新，不追加每一轮原始对话。
    memory_profile_max_orders: int = Field(default=20, ge=1, le=20)
    memory_profile_max_preferences: int = Field(default=10, ge=1, le=20)
    memory_profile_summary_max_chars: int = Field(default=800, ge=100, le=1000)
    memory_profile_max_bytes: int = Field(default=8192, ge=2048, le=65_536)
    # 0表示不按年龄删除；清理必须显式执行维护命令。
    memory_profile_retention_days: int = Field(default=0, ge=0, le=3650)
    memory_checkpoint_keep_last: int = Field(default=20, ge=2, le=1000)

    # RecursiveCharacterTextSplitter measures characters unless a token length function is set.
    rag_chunk_size: int = Field(default=700, ge=100, le=10_000)
    rag_chunk_overlap: int = Field(default=100, ge=0, le=2_000)
    rag_top_k: int = Field(default=3, ge=1, le=20)
    rag_min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    # 默认关键词3条 + 语义3条，交给聊天模型判断证据。
    rag_branch_top_k: int = Field(default=3, ge=1, le=10)
    rag_semantic_min_score: float = Field(default=0.35, ge=-1.0, le=1.0)
    rag_local_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    rag_retrieval_workers: int = Field(default=4, ge=2, le=16)
    # 独立排序模型，不复用聊天/embedding的接口地址或密钥。
    rag_rerank_enabled: bool = True
    rag_rerank_api_url: str | None = None
    # dashscope：百炼原生 text-rerank 协议；jina：Jina/Cohere 兼容的 /rerank 协议，
    # one-api、new-api 等 OpenAI-compatible 网关通常采用这种格式。
    rag_rerank_api_format: Literal["dashscope", "jina"] = "dashscope"
    rag_rerank_api_key: SecretStr | None = None
    rag_rerank_model_name: str | None = None
    rag_rerank_top_n: int = Field(default=3, ge=1, le=20)
    rag_rerank_max_candidates: int = Field(default=20, ge=1, le=50)
    rag_rerank_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    rag_rerank_max_document_chars: int = Field(default=2000, ge=100, le=8000)
    rag_rerank_min_score: float = Field(default=0.0, ge=0, le=1)
    rag_rrf_k: int = Field(default=60, ge=1, le=1000)
    rag_web_fallback_enabled: bool = True
    rag_web_mcp_transport: Literal["stdio", "streamable_http"] = "stdio"
    rag_web_mcp_command: str | None = "npx"
    rag_web_mcp_args: list[str] = Field(
        default_factory=lambda: ["-y", "@playwright/mcp@0.0.83", "--headless", "--isolated"]
    )
    rag_web_mcp_cache_dir: Path = Path("data/mcp_npm_cache")
    rag_web_mcp_url: str | None = None
    rag_web_mcp_token: SecretStr | None = None
    rag_web_search_url: str = "https://www.bing.com/search"
    rag_web_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    rag_web_max_chars: int = Field(default=12_000, ge=1000, le=30_000)

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65_535)
    admin_api_key: SecretStr | None = None

    @property
    def resolved_chat_api_key(self) -> SecretStr | None:
        return self.chat_api_key or self.openai_api_key

    @property
    def resolved_chat_base_url(self) -> str | None:
        return self.chat_base_url or self.openai_base_url

    @property
    def resolved_chat_model_name(self) -> str | None:
        return self.chat_model_name or self.openai_model_name

    @property
    def resolved_embedding_api_key(self) -> SecretStr | None:
        return self.embedding_api_key or self.openai_api_key

    @property
    def resolved_embedding_base_url(self) -> str | None:
        return self.embedding_base_url or self.openai_base_url

    @property
    def resolved_embedding_model_name(self) -> str | None:
        return self.embedding_model_name or self.openai_embedding_model_name

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def memory_db_file(self) -> Path:
        return self.resolve_path(self.memory_db_path)

    @property
    def multi_agent_db_file(self) -> Path:
        return self.resolve_path(self.multi_agent_db_path)

    @property
    def after_sales_db_file(self) -> Path:
        return self.resolve_path(self.after_sales_db_path)

    @property
    def rag_db_file(self) -> Path:
        return self.resolve_path(self.rag_db_path)

    @property
    def profile_db_file(self) -> Path:
        return self.resolve_path(self.profile_db_path)

    @property
    def ticket_db_file(self) -> Path:
        return self.resolve_path(self.ticket_db_path)

    @property
    def knowledge_base_path(self) -> Path:
        return self.resolve_path(self.knowledge_base_dir)

    def ensure_local_dirs(self) -> None:
        self.memory_db_file.parent.mkdir(parents=True, exist_ok=True)
        self.multi_agent_db_file.parent.mkdir(parents=True, exist_ok=True)
        self.after_sales_db_file.parent.mkdir(parents=True, exist_ok=True)
        self.profile_db_file.parent.mkdir(parents=True, exist_ok=True)
        self.rag_db_file.parent.mkdir(parents=True, exist_ok=True)
        self.ticket_db_file.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_local_dirs()
    return settings
