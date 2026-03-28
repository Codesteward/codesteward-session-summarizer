from __future__ import annotations

from datetime import UTC, datetime

import httpx
import structlog

from summarizer.config import Settings
from summarizer.models import SessionSummary

logger = structlog.get_logger()


class ClickHouseClient:
    """Async client for ClickHouse HTTP interface."""

    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.clickhouse_url.rstrip("/")
        self.database = settings.clickhouse_database
        self.params = {
            "database": self.database,
            "user": settings.clickhouse_user,
            "password": settings.clickhouse_password,
        }

    async def _query(self, sql: str, params: dict | None = None) -> list[dict]:
        """Execute a ClickHouse query and return rows as dicts."""
        query_params = {**self.params}
        if params:
            for k, v in params.items():
                query_params[f"param_{k}"] = str(v)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.base_url,
                params=query_params,
                content=sql + " FORMAT JSON",
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("data", [])

    async def _execute(self, sql: str) -> None:
        """Execute a ClickHouse statement (no result expected)."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.base_url,
                params=self.params,
                content=sql,
            )
            resp.raise_for_status()

    async def get_unsummarized_sessions(
        self,
        lookback_hours: int,
        cooldown_minutes: int,
        batch_size: int,
        summarizer_version: str,
    ) -> list[str]:
        """Find sessions that need summarization."""
        sql = """\
SELECT DISTINCT session_id
FROM audit_events
WHERE ts > now() - INTERVAL {lookback_hours:UInt32} HOUR
    AND session_id NOT IN (
        SELECT session_id FROM session_summaries FINAL
        WHERE summarizer_version = {current_version:String}
    )
    AND session_id NOT IN (
        SELECT session_id FROM audit_events
        WHERE ts > now() - INTERVAL {cooldown_minutes:UInt32} MINUTE
    )
ORDER BY max(ts) DESC
LIMIT {batch_size:UInt32}"""

        rows = await self._query(
            sql,
            params={
                "lookback_hours": lookback_hours,
                "cooldown_minutes": cooldown_minutes,
                "batch_size": batch_size,
                "current_version": summarizer_version,
            },
        )
        return [row["session_id"] for row in rows]

    async def get_session_events(self, session_id: str) -> list[dict]:
        """Load all events for a session, ordered by timestamp."""
        sql = """\
SELECT
    ts,
    direction,
    tool_name,
    substring(tool_input, 1, 1000) AS tool_input,
    assistant_text,
    thinking,
    model,
    agent,
    project,
    branch,
    user,
    team,
    input_tokens,
    output_tokens
FROM audit_events
WHERE session_id = {session_id:String}
ORDER BY ts ASC"""

        return await self._query(sql, params={"session_id": session_id})

    async def write_summary(self, summary: SessionSummary) -> None:
        """Write a session summary to ClickHouse."""

        # Format arrays for ClickHouse
        def fmt_array(arr: list[str]) -> str:
            escaped = [v.replace("'", "\\'") for v in arr]
            return "[" + ",".join(f"'{v}'" for v in escaped) + "]"

        def fmt_ts(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        sql = f"""\
INSERT INTO session_summaries (
    session_id, project, agent, branch, user, team,
    first_ts, last_ts, duration_minutes,
    turn_count, tool_call_count,
    total_input_tokens, total_output_tokens,
    summary, key_decisions, files_modified, tools_used, tags,
    summarizer_model, summarized_at, summarizer_version
) VALUES (
    '{summary.session_id}',
    '{summary.project}',
    '{summary.agent}',
    '{summary.branch}',
    '{summary.user}',
    '{summary.team}',
    '{fmt_ts(summary.first_ts)}',
    '{fmt_ts(summary.last_ts)}',
    {summary.duration_minutes},
    {summary.turn_count},
    {summary.tool_call_count},
    {summary.total_input_tokens},
    {summary.total_output_tokens},
    '{summary.summary.replace("'", "\\'")}',
    {fmt_array(summary.key_decisions)},
    {fmt_array(summary.files_modified)},
    {fmt_array(summary.tools_used)},
    {fmt_array(summary.tags)},
    '{summary.summarizer_model}',
    '{fmt_ts(summary.summarized_at)}',
    '{summary.summarizer_version}'
)"""

        await self._execute(sql)
        logger.info("session_summary_written", session_id=summary.session_id)


def compute_session_stats(events: list[dict]) -> dict:
    """Compute aggregate statistics from session events."""
    if not events:
        return {
            "first_ts": datetime.now(tz=UTC),
            "last_ts": datetime.now(tz=UTC),
            "duration_minutes": 0,
            "turn_count": 0,
            "tool_call_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "project": "",
            "agent": "",
            "branch": "",
            "user": "",
            "team": "",
        }

    timestamps = []
    for ev in events:
        ts = ev.get("ts", "")
        if isinstance(ts, str) and ts:
            try:
                # ClickHouse returns ISO-ish format
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(dt)
            except ValueError:
                pass
        elif isinstance(ts, datetime):
            timestamps.append(ts)

    first_ts = min(timestamps) if timestamps else datetime.now(tz=UTC)
    last_ts = max(timestamps) if timestamps else datetime.now(tz=UTC)
    duration_minutes = int((last_ts - first_ts).total_seconds() / 60)

    tool_call_count = sum(1 for ev in events if ev.get("tool_name"))

    total_input = sum(int(ev.get("input_tokens", 0) or 0) for ev in events)
    total_output = sum(int(ev.get("output_tokens", 0) or 0) for ev in events)

    # Use metadata from first event
    first = events[0]

    return {
        "first_ts": first_ts,
        "last_ts": last_ts,
        "duration_minutes": duration_minutes,
        "turn_count": len(events),
        "tool_call_count": tool_call_count,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "project": first.get("project", ""),
        "agent": first.get("agent", ""),
        "branch": first.get("branch", ""),
        "user": first.get("user", ""),
        "team": first.get("team", ""),
    }
