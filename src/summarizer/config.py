from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ClickHouse
    clickhouse_url: str = "http://localhost:8123"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_database: str = "audit"

    # LLM provider
    llm_provider: str = "ollama"  # "ollama", "openai", or "anthropic"
    summarizer_model: str = "phi3:mini"

    # Ollama
    ollama_url: str = "http://localhost:11434"

    # OpenAI (used when llm_provider=openai)
    openai_api_key: str = ""
    openai_base_url: str | None = None  # For OpenAI-compatible endpoints

    # Anthropic (used when llm_provider=anthropic)
    anthropic_api_key: str = ""

    # Processing
    poll_interval_seconds: int = 300
    session_cooldown_minutes: int = 30
    lookback_hours: int = 168
    batch_size: int = 10
    context_max_tokens: int = 4096
    context_max_chars: int | None = None  # Manual override — skips token calculation if set
    session_language: str = "en"
    summarizer_version: str = "v1"

    # Logging
    log_level: str = "info"
