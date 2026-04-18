from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    rabbitmq_url: str = Field(
        default="amqp://guest:guest@rabbitmq:5672/",
        description="AMQP connection URL for the RabbitMQ broker.",
    )
    tasks_queue: str = "agent-tasks-queue"
    results_queue: str = "agent-results-queue"
    errors_queue: str = "agent-errors-queue"

    ollama_base_url: str = "http://ollama:11434"
    default_model: str = "llama3.1:8b"
    default_temperature: float = 0.2
    llm_timeout_seconds: float = 120.0

    log_level: str = "INFO"
    prefetch_count: int = 1


settings = Settings()
