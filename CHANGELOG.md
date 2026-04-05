# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-04-05

### Fixed

- **structlog API compatibility** — replaced `structlog.get_level_from_name()` with `logging.getLevelName()` from the stdlib for compatibility with structlog ≥ 25.x which removed that function

## [0.4.0] - 2026-03-30

### Added
- Prompt provenance tracking: every summary and chunk extraction is tagged with `prompt_id`, `prompt_hash`, and `input_context_hash`
- Database-driven prompt loading (`PROMPT_SOURCE=database`) with automatic fallback to code defaults when no active prompt exists
- Optional evaluation data collection (`EVALUATION_ENABLED=true`) storing full input contexts in TTL-managed ClickHouse tables for downstream quality evaluation
- Prompt registry table (`prompt_registry`) for managing prompt versions with lineage tracking
- `hashing.py` utility for deterministic SHA-256 hash computation (truncated to 16 hex chars)

## [0.3.0] - 2026-03-28

### Added
- Core summarization pipeline: poll ClickHouse for unsummarized sessions, build token-efficient context, summarize via LLM, write results back
- Chunked extract-merge-synthesize pipeline for long sessions that exceed the LLM's context window
- 11 structured fact extraction categories: files_changed, decisions, constraints, bugs_resolved, tradeoffs, dependencies_changed, errors_encountered, test_actions, security_relevant, rollback_risks, boundaries
- Multi-provider LLM support: Ollama (local, default), OpenAI, and Anthropic via official SDKs
- Language-aware token budget calculator (28 languages) for optimal context window usage
- Smart chunking at natural time gap boundaries (>5 min gaps preferred over arbitrary splits)
- Secret stripping from all text before it reaches the LLM
- Auto-pull of Ollama models on startup
- Revision-based summary history: re-summarization creates a new revision instead of overwriting, preserving all prior summaries and chunk extractions
- Automatic detection of resumed sessions: sessions with new events after a prior summary are re-summarized on the next cycle
- `RUN_MODE=once` for single-run cron-style execution (in addition to continuous polling)
- Version-gated re-summarization via SUMMARIZER_VERSION
- Docker Compose setup with Ollama sidecar
- CI and release GitHub Actions workflows
- ClickHouse migrations for session_summaries and session_chunk_extractions tables
