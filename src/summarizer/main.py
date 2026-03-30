from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from summarizer.clickhouse import ClickHouseClient, compute_session_stats
from summarizer.config import Settings
from summarizer.context_builder import (
    EXTRACTION_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_extraction_prompt,
    build_prompt_context,
    build_synthesis_prompt,
    build_user_prompt,
    chunk_events,
    extract_files_from_events,
    extract_tools_from_events,
    merge_extractions,
    needs_chunked_processing,
)
from summarizer.hashing import compute_hash
from summarizer.llm import LLMClient, create_llm_client
from summarizer.models import (
    ChunkEvaluationContext,
    ChunkExtraction,
    SessionSummary,
    SummaryEvaluationContext,
)
from summarizer.token_budget import calculate_char_budget

logger = structlog.get_logger()

MIN_SESSION_EVENTS = 3

# Default prompt IDs used when prompts come from code
_CODE_DEFAULT_PROMPT_ID = ""

# Maps prompt roles to their code default constants
_CODE_DEFAULTS: dict[str, str] = {
    "extraction": EXTRACTION_SYSTEM_PROMPT,
    "synthesis": SYNTHESIS_SYSTEM_PROMPT,
    "summary": SYSTEM_PROMPT,
}


@dataclass
class PromptInfo:
    """Holds a prompt's text and provenance metadata."""

    prompt_id: str
    prompt_text: str
    prompt_hash: str


async def _llm_call_with_retry(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    session_id: str,
    label: str,
    *,
    extraction: bool = False,
) -> dict | None:
    """Call LLM with one retry. Returns parsed response or None on failure.

    If extraction=True, uses generate_extraction() for fact-category parsing.
    """
    method = llm.generate_extraction if extraction else llm.generate
    for attempt in range(2):
        try:
            return await method(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning(
                "llm_call_failed",
                session_id=session_id,
                label=label,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt == 1:
                logger.error(
                    "session_summary_failed",
                    session_id=session_id,
                    error=f"LLM {label} failed after 2 attempts: {exc}",
                )
                return None
    return None


async def _extract_and_persist_facts(
    session_id: str,
    revision: int,
    events: list[dict],
    char_budget: int,
    ch: ClickHouseClient,
    llm: LLMClient,
    settings: Settings,
    extraction_prompt_info: PromptInfo,
    *,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> dict | None:
    """Extract structured facts from events and persist as a chunk extraction.

    Returns the extraction dict on success, or None on LLM failure.
    """
    context_text = build_prompt_context(events, max_chars=char_budget)
    extraction_prompt = build_extraction_prompt(
        session_id=session_id,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        context_text=context_text,
    )

    extraction = await _llm_call_with_retry(
        llm,
        extraction_prompt_info.prompt_text,
        extraction_prompt,
        session_id,
        f"extract_chunk_{chunk_index}",
        extraction=True,
    )
    if extraction is None:
        return None

    input_context_hash = compute_hash(extraction_prompt)

    chunk_stats = compute_session_stats(events)
    chunk_extraction = ChunkExtraction(
        session_id=session_id,
        revision=revision,
        chunk_index=chunk_index,
        chunk_start_ts=chunk_stats["first_ts"],
        chunk_end_ts=chunk_stats["last_ts"],
        event_count=len(events),
        files_changed=extraction.get("files_changed", []),
        decisions=extraction.get("decisions", []),
        constraints=extraction.get("constraints", []),
        bugs_resolved=extraction.get("bugs_resolved", []),
        tradeoffs=extraction.get("tradeoffs", []),
        dependencies_changed=extraction.get("dependencies_changed", []),
        errors_encountered=extraction.get("errors_encountered", []),
        test_actions=extraction.get("test_actions", []),
        security_relevant=extraction.get("security_relevant", []),
        rollback_risks=extraction.get("rollback_risks", []),
        boundaries=extraction.get("boundaries", []),
        summarizer_model=settings.summarizer_model,
        summarizer_version=settings.summarizer_version,
        extracted_at=datetime.now(tz=UTC),
        prompt_id=extraction_prompt_info.prompt_id,
        prompt_hash=extraction_prompt_info.prompt_hash,
        input_context_hash=input_context_hash,
    )

    try:
        await ch.write_chunk_extraction(chunk_extraction)
    except Exception as exc:
        logger.error(
            "chunk_extraction_write_failed",
            session_id=session_id,
            chunk_index=chunk_index,
            error=str(exc),
        )
        # Continue — extraction is persisted best-effort

    # Store evaluation context if enabled
    if settings.evaluation_enabled:
        try:
            await ch.write_chunk_evaluation_context(
                ChunkEvaluationContext(
                    session_id=session_id,
                    revision=revision,
                    chunk_index=chunk_index,
                    prompt_id=extraction_prompt_info.prompt_id,
                    prompt_hash=extraction_prompt_info.prompt_hash,
                    input_context_hash=input_context_hash,
                    input_context=extraction_prompt,
                    stored_at=datetime.now(tz=UTC),
                )
            )
        except Exception as exc:
            logger.warning(
                "chunk_evaluation_context_write_failed",
                session_id=session_id,
                chunk_index=chunk_index,
                error=str(exc),
            )

    return extraction


async def _process_single_pass(
    session_id: str,
    revision: int,
    events: list[dict],
    stats: dict,
    files: list[str],
    tools: list[str],
    char_budget: int,
    ch: ClickHouseClient,
    llm: LLMClient,
    settings: Settings,
    prompts: dict[str, PromptInfo],
) -> bool:
    """Single-pass summarization for short sessions."""
    context_text = build_prompt_context(events, max_chars=char_budget)
    logger.info(
        "session_summarizing",
        session_id=session_id,
        event_count=len(events),
        context_chars=len(context_text),
        mode="single_pass",
    )

    # Extract structured facts (persisted as chunk_index=0)
    extraction = await _extract_and_persist_facts(
        session_id,
        revision,
        events,
        char_budget,
        ch,
        llm,
        settings,
        prompts["extraction"],
    )
    if extraction is None:
        return False

    summary_prompt_info = prompts["summary"]
    user_prompt = build_user_prompt(
        session_id=session_id,
        project=stats["project"],
        agent=stats["agent"],
        branch=stats["branch"],
        duration_minutes=stats["duration_minutes"],
        tools=tools,
        files=files,
        context_text=context_text,
    )

    start_time = asyncio.get_event_loop().time()
    llm_result = await _llm_call_with_retry(
        llm, summary_prompt_info.prompt_text, user_prompt, session_id, "summarize"
    )
    if llm_result is None:
        return False
    llm_duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

    input_context_hash = compute_hash(user_prompt)

    # Store evaluation context if enabled
    if settings.evaluation_enabled:
        try:
            await ch.write_summary_evaluation_context(
                SummaryEvaluationContext(
                    session_id=session_id,
                    revision=revision,
                    prompt_id=summary_prompt_info.prompt_id,
                    prompt_hash=summary_prompt_info.prompt_hash,
                    input_context_hash=input_context_hash,
                    input_context=user_prompt,
                    stored_at=datetime.now(tz=UTC),
                )
            )
        except Exception as exc:
            logger.warning(
                "summary_evaluation_context_write_failed",
                session_id=session_id,
                error=str(exc),
            )

    return await _write_summary(
        session_id,
        revision,
        stats,
        files,
        tools,
        llm_result,
        llm_duration_ms,
        ch,
        settings,
        summary_prompt_info,
        input_context_hash,
    )


async def _process_chunked(
    session_id: str,
    revision: int,
    events: list[dict],
    stats: dict,
    files: list[str],
    tools: list[str],
    char_budget: int,
    ch: ClickHouseClient,
    llm: LLMClient,
    settings: Settings,
    prompts: dict[str, PromptInfo],
) -> bool:
    """Chunked extract-merge-synthesize for long sessions."""
    chunks = chunk_events(events, max_chars=char_budget)
    logger.info(
        "session_summarizing",
        session_id=session_id,
        revision=revision,
        event_count=len(events),
        mode="chunked",
        chunk_count=len(chunks),
    )

    # Step 1: Extract facts from each chunk
    extractions: list[dict] = []
    for i, chunk in enumerate(chunks):
        extraction = await _extract_and_persist_facts(
            session_id,
            revision,
            chunk,
            char_budget,
            ch,
            llm,
            settings,
            prompts["extraction"],
            chunk_index=i,
            total_chunks=len(chunks),
        )
        if extraction is None:
            return False
        extractions.append(extraction)

    # Step 2: Merge all extracted facts
    merged_facts = merge_extractions(extractions)

    # Step 3: Synthesize final summary
    synthesis_prompt_info = prompts["synthesis"]
    synthesis_prompt = build_synthesis_prompt(
        session_id=session_id,
        project=stats["project"],
        agent=stats["agent"],
        branch=stats["branch"],
        duration_minutes=stats["duration_minutes"],
        tools=tools,
        files=files,
        merged_facts=merged_facts,
    )

    start_time = asyncio.get_event_loop().time()
    llm_result = await _llm_call_with_retry(
        llm, synthesis_prompt_info.prompt_text, synthesis_prompt, session_id, "synthesize"
    )
    if llm_result is None:
        return False
    llm_duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

    input_context_hash = compute_hash(synthesis_prompt)

    # Store evaluation context if enabled
    if settings.evaluation_enabled:
        try:
            await ch.write_summary_evaluation_context(
                SummaryEvaluationContext(
                    session_id=session_id,
                    revision=revision,
                    prompt_id=synthesis_prompt_info.prompt_id,
                    prompt_hash=synthesis_prompt_info.prompt_hash,
                    input_context_hash=input_context_hash,
                    input_context=synthesis_prompt,
                    stored_at=datetime.now(tz=UTC),
                )
            )
        except Exception as exc:
            logger.warning(
                "summary_evaluation_context_write_failed",
                session_id=session_id,
                error=str(exc),
            )

    return await _write_summary(
        session_id,
        revision,
        stats,
        files,
        tools,
        llm_result,
        llm_duration_ms,
        ch,
        settings,
        synthesis_prompt_info,
        input_context_hash,
    )


async def _write_summary(
    session_id: str,
    revision: int,
    stats: dict,
    files: list[str],
    tools: list[str],
    llm_result: dict,
    llm_duration_ms: int,
    ch: ClickHouseClient,
    settings: Settings,
    prompt_info: PromptInfo,
    input_context_hash: str,
) -> bool:
    """Build and write a SessionSummary to ClickHouse."""
    summary = SessionSummary(
        session_id=session_id,
        revision=revision,
        project=stats["project"],
        agent=stats["agent"],
        branch=stats["branch"],
        user=stats["user"],
        team=stats["team"],
        first_ts=stats["first_ts"],
        last_ts=stats["last_ts"],
        duration_minutes=stats["duration_minutes"],
        turn_count=stats["turn_count"],
        tool_call_count=stats["tool_call_count"],
        total_input_tokens=stats["total_input_tokens"],
        total_output_tokens=stats["total_output_tokens"],
        summary=llm_result.get("summary", ""),
        key_decisions=llm_result.get("key_decisions", []),
        files_modified=files,
        tools_used=tools,
        tags=llm_result.get("tags", []),
        summarizer_model=settings.summarizer_model,
        summarized_at=datetime.now(tz=UTC),
        summarizer_version=settings.summarizer_version,
        prompt_id=prompt_info.prompt_id,
        prompt_hash=prompt_info.prompt_hash,
        input_context_hash=input_context_hash,
    )

    try:
        await ch.write_summary(summary)
    except Exception as exc:
        logger.error("session_summary_failed", session_id=session_id, error=str(exc))
        return False

    logger.info(
        "session_summarized",
        session_id=session_id,
        revision=revision,
        summary_length=len(summary.summary),
        tags=summary.tags,
        llm_duration_ms=llm_duration_ms,
    )
    return True


def _resolve_char_budget(settings: Settings) -> int:
    """Resolve the character budget from settings.

    If CONTEXT_MAX_CHARS is set explicitly, use that (manual override).
    Otherwise, calculate from CONTEXT_MAX_TOKENS + SESSION_LANGUAGE.
    """
    if settings.context_max_chars is not None:
        return settings.context_max_chars
    return calculate_char_budget(
        max_tokens=settings.context_max_tokens,
        language=settings.session_language,
    )


async def _load_prompts(ch: ClickHouseClient, settings: Settings) -> dict[str, PromptInfo]:
    """Load prompts for all roles based on PROMPT_SOURCE setting.

    When PROMPT_SOURCE=database, tries to load from prompt_registry.
    Falls back to code defaults for any role without an active DB prompt.
    """
    prompts: dict[str, PromptInfo] = {}

    for role, default_text in _CODE_DEFAULTS.items():
        if settings.prompt_source == "database":
            try:
                db_prompt = await ch.get_active_prompt(role)
            except Exception as exc:
                logger.warning(
                    "prompt_registry_read_failed",
                    role=role,
                    error=str(exc),
                )
                db_prompt = None

            if db_prompt:
                prompts[role] = PromptInfo(
                    prompt_id=db_prompt["prompt_id"],
                    prompt_text=db_prompt["prompt_text"],
                    prompt_hash=db_prompt["prompt_hash"],
                )
                logger.info(
                    "prompt_loaded_from_db",
                    role=role,
                    prompt_id=db_prompt["prompt_id"],
                )
                continue

            logger.info(
                "prompt_fallback_to_code",
                role=role,
                reason="no active prompt in registry",
            )

        prompts[role] = PromptInfo(
            prompt_id=_CODE_DEFAULT_PROMPT_ID,
            prompt_text=default_text,
            prompt_hash=compute_hash(default_text),
        )

    return prompts


async def process_session(
    session_id: str,
    ch: ClickHouseClient,
    llm: LLMClient,
    settings: Settings,
    prompts: dict[str, PromptInfo],
) -> bool:
    """Process a single session: load events, summarize, write result.

    Uses single-pass for short sessions, chunked extract-merge-synthesize for long ones.
    Returns True if successful, False otherwise.
    """
    events = await ch.get_session_events(session_id)

    if len(events) < MIN_SESSION_EVENTS:
        logger.info(
            "session_skipped",
            session_id=session_id,
            reason="too_few_events",
            event_count=len(events),
        )
        return False

    stats = compute_session_stats(events)
    files = extract_files_from_events(events)
    tools = extract_tools_from_events(events)
    char_budget = _resolve_char_budget(settings)
    revision = await ch.get_next_revision(session_id)

    if needs_chunked_processing(events, max_chars=char_budget):
        return await _process_chunked(
            session_id,
            revision,
            events,
            stats,
            files,
            tools,
            char_budget,
            ch,
            llm,
            settings,
            prompts,
        )
    else:
        return await _process_single_pass(
            session_id,
            revision,
            events,
            stats,
            files,
            tools,
            char_budget,
            ch,
            llm,
            settings,
            prompts,
        )


async def poll_cycle(
    ch: ClickHouseClient,
    llm: LLMClient,
    settings: Settings,
    prompts: dict[str, PromptInfo],
) -> None:
    """Run one polling cycle: discover and process unsummarized sessions."""
    logger.info(
        "poll_started",
        lookback_hours=settings.lookback_hours,
        batch_size=settings.batch_size,
    )

    try:
        session_ids = await ch.get_unsummarized_sessions(
            lookback_hours=settings.lookback_hours,
            cooldown_minutes=settings.session_cooldown_minutes,
            batch_size=settings.batch_size,
            summarizer_version=settings.summarizer_version,
        )
    except Exception as exc:
        logger.error("poll_discovery_failed", error=str(exc))
        return

    logger.info("sessions_found", count=len(session_ids))

    for session_id in session_ids:
        await process_session(session_id, ch, llm, settings, prompts)


async def run() -> None:
    """Main entry point: configure, check health, and start processing.

    Supports two run modes:
    - "poll" (default): continuous polling loop, runs forever
    - "once": single cycle, then exit — suitable for cron/scheduled jobs
    """
    settings = Settings()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(settings.log_level)
        ),
    )

    run_mode = settings.run_mode.lower().strip()
    if run_mode not in ("poll", "once"):
        logger.critical("invalid_run_mode", run_mode=run_mode)
        sys.exit(1)

    char_budget = _resolve_char_budget(settings)
    logger.info(
        "summarizer_starting",
        provider=settings.llm_provider,
        model=settings.summarizer_model,
        run_mode=run_mode,
        poll_interval=settings.poll_interval_seconds if run_mode == "poll" else None,
        version=settings.summarizer_version,
        context_max_tokens=settings.context_max_tokens,
        session_language=settings.session_language,
        char_budget=char_budget,
        prompt_source=settings.prompt_source,
        evaluation_enabled=settings.evaluation_enabled,
    )

    llm = create_llm_client(settings)

    # Ensure model is available (pull if needed for Ollama)
    if not await llm.ensure_model():
        logger.critical(
            "summarizer_startup_failed",
            reason="model_not_available",
            model=settings.summarizer_model,
        )
        sys.exit(1)

    ch = ClickHouseClient(settings)

    # Load prompts (from DB or code defaults)
    prompts = await _load_prompts(ch, settings)
    for role, info in prompts.items():
        logger.info(
            "prompt_active",
            role=role,
            prompt_id=info.prompt_id or "(code default)",
            prompt_hash=info.prompt_hash,
        )

    if run_mode == "once":
        await poll_cycle(ch, llm, settings, prompts)
        logger.info("summarizer_finished", run_mode="once")
        return

    while True:
        await poll_cycle(ch, llm, settings, prompts)
        logger.debug("poll_sleeping", seconds=settings.poll_interval_seconds)
        await asyncio.sleep(settings.poll_interval_seconds)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
