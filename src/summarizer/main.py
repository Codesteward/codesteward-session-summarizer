from __future__ import annotations

import asyncio
import sys
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
from summarizer.llm import LLMClient, create_llm_client
from summarizer.models import ChunkExtraction, SessionSummary
from summarizer.token_budget import calculate_char_budget

logger = structlog.get_logger()

MIN_SESSION_EVENTS = 3


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


async def _process_single_pass(
    session_id: str,
    events: list[dict],
    stats: dict,
    files: list[str],
    tools: list[str],
    char_budget: int,
    ch: ClickHouseClient,
    llm: LLMClient,
    settings: Settings,
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
        llm, SYSTEM_PROMPT, user_prompt, session_id, "summarize"
    )
    if llm_result is None:
        return False
    llm_duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

    return await _write_summary(
        session_id, stats, files, tools, llm_result, llm_duration_ms, ch, settings
    )


async def _process_chunked(
    session_id: str,
    events: list[dict],
    stats: dict,
    files: list[str],
    tools: list[str],
    char_budget: int,
    ch: ClickHouseClient,
    llm: LLMClient,
    settings: Settings,
) -> bool:
    """Chunked extract-merge-synthesize for long sessions."""
    chunks = chunk_events(events, max_chars=char_budget)
    logger.info(
        "session_summarizing",
        session_id=session_id,
        event_count=len(events),
        mode="chunked",
        chunk_count=len(chunks),
    )

    # Step 1: Extract facts from each chunk
    extractions: list[dict] = []
    for i, chunk in enumerate(chunks):
        context_text = build_prompt_context(chunk, max_chars=char_budget)
        extraction_prompt = build_extraction_prompt(
            session_id=session_id,
            chunk_index=i,
            total_chunks=len(chunks),
            context_text=context_text,
        )

        extraction = await _llm_call_with_retry(
            llm,
            EXTRACTION_SYSTEM_PROMPT,
            extraction_prompt,
            session_id,
            f"extract_chunk_{i}",
            extraction=True,
        )
        if extraction is None:
            return False
        extractions.append(extraction)

        # Write chunk extraction to ClickHouse
        chunk_stats = compute_session_stats(chunk)
        chunk_extraction = ChunkExtraction(
            session_id=session_id,
            chunk_index=i,
            chunk_start_ts=chunk_stats["first_ts"],
            chunk_end_ts=chunk_stats["last_ts"],
            event_count=len(chunk),
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
        )

        try:
            await ch.write_chunk_extraction(chunk_extraction)
        except Exception as exc:
            logger.error(
                "chunk_extraction_write_failed",
                session_id=session_id,
                chunk_index=i,
                error=str(exc),
            )
            # Continue — extraction is persisted best-effort

    # Step 2: Merge all extracted facts
    merged_facts = merge_extractions(extractions)

    # Step 3: Synthesize final summary
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
        llm, SYNTHESIS_SYSTEM_PROMPT, synthesis_prompt, session_id, "synthesize"
    )
    if llm_result is None:
        return False
    llm_duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

    return await _write_summary(
        session_id, stats, files, tools, llm_result, llm_duration_ms, ch, settings
    )


async def _write_summary(
    session_id: str,
    stats: dict,
    files: list[str],
    tools: list[str],
    llm_result: dict,
    llm_duration_ms: int,
    ch: ClickHouseClient,
    settings: Settings,
) -> bool:
    """Build and write a SessionSummary to ClickHouse."""
    summary = SessionSummary(
        session_id=session_id,
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
    )

    try:
        await ch.write_summary(summary)
    except Exception as exc:
        logger.error("session_summary_failed", session_id=session_id, error=str(exc))
        return False

    logger.info(
        "session_summarized",
        session_id=session_id,
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


async def process_session(
    session_id: str,
    ch: ClickHouseClient,
    llm: LLMClient,
    settings: Settings,
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

    if needs_chunked_processing(events, max_chars=char_budget):
        return await _process_chunked(
            session_id, events, stats, files, tools, char_budget, ch, llm, settings
        )
    else:
        return await _process_single_pass(
            session_id, events, stats, files, tools, char_budget, ch, llm, settings
        )


async def poll_cycle(ch: ClickHouseClient, llm: LLMClient, settings: Settings) -> None:
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
        await process_session(session_id, ch, llm, settings)


async def run() -> None:
    """Main entry point: configure, check health, and start polling loop."""
    settings = Settings()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(settings.log_level)
        ),
    )

    char_budget = _resolve_char_budget(settings)
    logger.info(
        "summarizer_starting",
        provider=settings.llm_provider,
        model=settings.summarizer_model,
        poll_interval=settings.poll_interval_seconds,
        version=settings.summarizer_version,
        context_max_tokens=settings.context_max_tokens,
        session_language=settings.session_language,
        char_budget=char_budget,
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

    while True:
        await poll_cycle(ch, llm, settings)
        logger.debug("poll_sleeping", seconds=settings.poll_interval_seconds)
        await asyncio.sleep(settings.poll_interval_seconds)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
