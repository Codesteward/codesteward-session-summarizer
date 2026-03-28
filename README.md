# codesteward-session-summarizer

Background service that reads raw audit events from ClickHouse, summarizes development sessions using an LLM, and writes structured summaries back to ClickHouse. Supports [Ollama](https://ollama.com) (local), OpenAI, and Anthropic as LLM providers.

Summaries are served by the `session_summaries` MCP tool in `codesteward-mcp`.

## How it works

1. **Poll** ClickHouse for sessions with no activity in the last 30 minutes (configurable)
2. **Build** a token-efficient context from audit events (timestamps, tool names, file paths, assistant text snippets)
3. **Summarize** via an LLM (Ollama, OpenAI, or Anthropic) — produces a natural language summary, key decisions, and topic tags
4. **Write** the structured summary back to ClickHouse (`session_summaries` table)

The summarizer is idempotent: re-running produces the same results. Bumping `SUMMARIZER_VERSION` triggers re-summarization of all sessions.

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- ClickHouse accessible via HTTP
- An LLM provider: [Ollama](https://ollama.com) running locally, or an OpenAI/Anthropic API key

### Run with Ollama (default)

```bash
uv sync
uv run python -m summarizer.main
```

The summarizer will auto-pull the configured model on first start.

### Run with OpenAI

```bash
uv sync --extra openai
LLM_PROVIDER=openai SUMMARIZER_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... uv run python -m summarizer.main
```

### Run with Anthropic

```bash
uv sync --extra anthropic
LLM_PROVIDER=anthropic SUMMARIZER_MODEL=claude-haiku-4-5-20251001 ANTHROPIC_API_KEY=sk-ant-... uv run python -m summarizer.main
```

### Run with Docker Compose

```bash
docker compose up -d
```

This starts both the summarizer and an Ollama sidecar. The model is pulled automatically.

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLICKHOUSE_URL` | `http://localhost:8123` | ClickHouse HTTP interface |
| `CLICKHOUSE_USER` | `default` | ClickHouse user |
| `CLICKHOUSE_PASSWORD` | `""` | ClickHouse password |
| `CLICKHOUSE_DATABASE` | `audit` | Database name |
| `LLM_PROVIDER` | `ollama` | LLM provider: `ollama`, `openai`, or `anthropic` |
| `SUMMARIZER_MODEL` | `phi3:mini` | Model name (provider-specific) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OPENAI_API_KEY` | `""` | OpenAI API key (required when provider=openai) |
| `OPENAI_BASE_URL` | *unset* | Custom OpenAI-compatible endpoint |
| `ANTHROPIC_API_KEY` | `""` | Anthropic API key (required when provider=anthropic) |
| `POLL_INTERVAL_SECONDS` | `300` | Seconds between polling cycles |
| `SESSION_COOLDOWN_MINUTES` | `30` | Minutes of inactivity before summarizing |
| `LOOKBACK_HOURS` | `168` | How far back to look (default: 7 days) |
| `BATCH_SIZE` | `10` | Max sessions per cycle |
| `CONTEXT_MAX_TOKENS` | `4096` | Model's context window in tokens |
| `SESSION_LANGUAGE` | `en` | Session language for token budget calculation |
| `CONTEXT_MAX_CHARS` | (calculated) | Manual override — skips token calculation if set |
| `SUMMARIZER_VERSION` | `v1` | Bump to force re-summarization |
| `LOG_LEVEL` | `info` | Logging level |

The character budget for LLM prompts is calculated automatically from `CONTEXT_MAX_TOKENS` and `SESSION_LANGUAGE`. For example, a 32K-token Mistral model with German sessions (`CONTEXT_MAX_TOKENS=32768 SESSION_LANGUAGE=de`) gets a ~94K character budget. Set `CONTEXT_MAX_CHARS` to override the calculation.

## Development

```bash
# Install all dependencies (including dev)
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## ClickHouse migration

Apply `migrations/008_session_summaries.sql` to create the `session_summaries` table. This uses `ReplacingMergeTree(summarized_at)` to keep only the latest summary per session.
