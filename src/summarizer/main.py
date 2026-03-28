from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

import structlog

from summarizer.clickhouse import ClickHouseClient, compute_session_stats
from summarizer.config import Settings
from summarizer.context_builder import (
    SYSTEM_PROMPT,
    build_prompt_context,
    build_user_prompt,
    extract_files_from_events,
    extract_tools_from_events,
)
from summarizer.llm import OllamaClient
from summarizer.models import SessionSummary

logger = structlog.get_logger()

MIN_SESSION_EVENTS = 3


async def process_session(
    session_id: str,
    ch: ClickHouseClient,
    ollama: OllamaClient,
    settings: Settings,
) -> bool:
    """Process a single session: load events, summarize, write result.

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

    context_text = build_prompt_context(events, max_chars=settings.context_max_chars)
    logger.info(
        "session_summarizing",
        session_id=session_id,
        event_count=len(events),
        context_chars=len(context_text),
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

    # Attempt LLM call with one retry
    llm_result = None
    start_time = asyncio.get_event_loop().time()
    for attempt in range(2):
        try:
            llm_result = await ollama.generate(SYSTEM_PROMPT, user_prompt)
            break
        except Exception as exc:
            logger.warning(
                "llm_call_failed",
                session_id=session_id,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt == 1:
                logger.error(
                    "session_summary_failed",
                    session_id=session_id,
                    error=f"LLM failed after 2 attempts: {exc}",
                )
                return False

    llm_duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

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
        summary=llm_result["summary"],
        key_decisions=llm_result["key_decisions"],
        files_modified=files,
        tools_used=tools,
        tags=llm_result["tags"],
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


async def poll_cycle(ch: ClickHouseClient, ollama: OllamaClient, settings: Settings) -> None:
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
        await process_session(session_id, ch, ollama, settings)


async def run() -> None:
    """Main entry point: configure, check health, and start polling loop."""
    settings = Settings()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(settings.log_level)
        ),
    )

    logger.info(
        "summarizer_starting",
        model=settings.summarizer_model,
        poll_interval=settings.poll_interval_seconds,
        version=settings.summarizer_version,
    )

    ollama = OllamaClient(settings.ollama_url, settings.summarizer_model)

    # Ensure model is available (pull if needed)
    if not await ollama.ensure_model():
        logger.critical(
            "summarizer_startup_failed",
            reason="model_not_available",
            model=settings.summarizer_model,
        )
        sys.exit(1)

    ch = ClickHouseClient(settings)

    while True:
        await poll_cycle(ch, ollama, settings)
        logger.debug("poll_sleeping", seconds=settings.poll_interval_seconds)
        await asyncio.sleep(settings.poll_interval_seconds)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
