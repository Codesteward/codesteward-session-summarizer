from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ClickHouse
    clickhouse_url: str = "http://localhost:8123"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "audit"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    summarizer_model: str = "phi3:mini"

    # Processing
    poll_interval_seconds: int = 300
    session_cooldown_minutes: int = 30
    lookback_hours: int = 168
    batch_size: int = 10
    context_max_chars: int = 6000
    summarizer_version: str = "v1"

    # Logging
    log_level: str = "info"
