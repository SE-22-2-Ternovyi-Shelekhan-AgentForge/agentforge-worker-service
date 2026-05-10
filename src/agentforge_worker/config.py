from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # RabbitMQ
    rabbitmq_url: str = Field(
        default="amqp://guest:guest@rabbitmq:5672/",
        description="AMQP connection URL for the RabbitMQ broker.",
    )
    sessions_queue: str = "agent-sessions-queue"
    events_queue: str = "agent-events-queue"
    results_queue: str = "agent-results-queue"
    errors_queue: str = "agent-errors-queue"

    # LLM provider abstraction
    default_provider: str = "ollama"        # "ollama" | "openai"
    supervisor_provider: str = "ollama"
    default_model: str = "qwen2.5:7b"
    default_temperature: float = 0.2
    supervisor_model: str = "qwen2.5:7b"
    llm_timeout_seconds: float = 180.0

    # Ollama-specific
    ollama_base_url: str = "http://ollama:11434"

    # OpenAI-specific (used only when provider == "openai")
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # Misc
    log_level: str = "INFO"
    prefetch_count: int = 1
    workspace_path: str = "/workspace"


settings = Settings()
