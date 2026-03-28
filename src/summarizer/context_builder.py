from __future__ import annotations

import json
import re


def extract_file_path(tool_input: str) -> str | None:
    """Extract a file path from a tool_input JSON string."""
    if not tool_input:
        return None
    try:
        data = json.loads(tool_input)
    except (json.JSONDecodeError, TypeError):
        return None

    for key in ("file_path", "path", "filename", "file"):
        if key in data and isinstance(data[key], str):
            return data[key]
    return None


def extract_files_from_events(events: list[dict]) -> list[str]:
    """Extract deduplicated file paths from all events."""
    seen: set[str] = set()
    paths: list[str] = []
    for ev in events:
        fp = extract_file_path(ev.get("tool_input", ""))
        if fp and fp not in seen:
            seen.add(fp)
            paths.append(fp)
    return paths


def extract_tools_from_events(events: list[dict]) -> list[str]:
    """Extract deduplicated tool names from all events."""
    seen: set[str] = set()
    tools: list[str] = []
    for ev in events:
        tool = ev.get("tool_name", "")
        if tool and tool not in seen:
            seen.add(tool)
            tools.append(tool)
    return tools


def _strip_secrets(text: str) -> str:
    """Remove potential secret values from text."""
    return re.sub(
        r"(password|secret|token|key|credential)\s*[=:]\s*\S+",
        r"\1=***",
        text,
        flags=re.IGNORECASE,
    )


def build_prompt_context(events: list[dict], max_chars: int = 6000) -> str:
    """Build a token-efficient session representation for the LLM.

    Priority order:
    1. Always: timestamps, tool names, file paths
    2. If budget allows: assistant text snippets (first 200 chars)
    3. If budget allows: thinking block snippets (first 100 chars)
    4. Never: raw tool inputs, API response bodies
    """
    lines: list[str] = []
    char_budget = max_chars

    for i, ev in enumerate(events):
        ts = ev.get("ts", "")
        tool = ev.get("tool_name", "")
        direction = ev.get("direction", "")

        if tool:
            file_path = extract_file_path(ev.get("tool_input", ""))
            line = f"[{ts}] {tool}"
            if file_path:
                line += f" → {file_path}"
        else:
            line = f"[{ts}] {direction}"

        lines.append(line)
        char_budget -= len(line) + 1  # +1 for newline

        # Assistant text snippet if budget allows
        if char_budget > 500:
            assistant_text = ev.get("assistant_text", "")
            if isinstance(assistant_text, list):
                for text in assistant_text:
                    snippet = _strip_secrets(str(text)[:200]).replace("\n", " ").strip()
                    if snippet:
                        entry = f"  said: {snippet}"
                        lines.append(entry)
                        char_budget -= len(entry) + 1
            elif isinstance(assistant_text, str) and assistant_text:
                snippet = _strip_secrets(assistant_text[:200]).replace("\n", " ").strip()
                if snippet:
                    entry = f"  said: {snippet}"
                    lines.append(entry)
                    char_budget -= len(entry) + 1

        # Thinking snippet if budget allows
        if char_budget > 300:
            thinking = ev.get("thinking", "")
            if isinstance(thinking, str) and thinking:
                snippet = _strip_secrets(thinking[:100]).replace("\n", " ").strip()
                if snippet:
                    entry = f"  thought: {snippet}"
                    lines.append(entry)
                    char_budget -= len(entry) + 1

        if char_budget <= 0:
            remaining = len(events) - i - 1
            if remaining > 0:
                lines.append(f"... ({remaining} more events)")
            break

    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a development session summarizer. Given a chronological log of \
tool calls and assistant messages from an AI coding session, produce a \
structured summary.

Respond in YAML format with these fields:
- summary: 2-4 sentences describing what was accomplished
- key_decisions: list of 3-7 bullet points (key changes, architectural decisions, problems solved)
- tags: list of 3-8 topic tags (e.g. "authentication", "refactoring", "bug-fix", "testing")

Focus on WHAT was accomplished and WHY, not the mechanical steps.
Highlight any significant decisions, trade-offs, or problems encountered.\
"""

EXTRACTION_SYSTEM_PROMPT = """\
You are a development session fact extractor. Given a chronological log of \
tool calls and assistant messages from a chunk of an AI coding session, \
extract ALL important facts into structured categories.

Be comprehensive and precise — these facts will be used for code reviews. \
Do not summarize or generalize. Capture specific file names, error messages, \
decision rationale, and concrete details.

Respond in YAML format with these fields:
- files_changed: list of "file/path — what changed" entries
- decisions: list of "decision — reasoning" entries
- constraints: list of problems, blockers, or limitations encountered
- bugs_resolved: list of "bug description — how it was resolved" entries
- tradeoffs: list of "what was skipped or deferred — why" entries
- dependencies_changed: list of "package added/removed/upgraded — why" entries
- errors_encountered: list of "error — resolution status (fixed/worked-around/unresolved)" entries
- test_actions: list of "test file/function — added/modified/removed/skipped" entries
- security_relevant: list of changes to auth, permissions, validation, or secrets handling
- rollback_risks: list of migrations, schema changes, or other hard-to-reverse changes
- boundaries: list of explicit rules, invariants, or must/must-not constraints \
defined or reinforced (e.g. "never modify audit_events directly", \
"API must stay backward compatible with v1")

If a category has no entries, use an empty list []. \
Never omit a category.\
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a development session summarizer. Given extracted facts from all \
chunks of an AI coding session, produce a final comprehensive summary.

These summaries are used for code reviews, so be precise and comprehensive. \
Do not drop important decisions, constraints, boundaries, or risks.

Respond in YAML format with these fields:
- summary: 2-4 sentences describing what was accomplished and why
- key_decisions: list of 5-10 bullet points covering all key changes, \
architectural decisions, problems solved, and trade-offs made
- tags: list of 3-8 topic tags (e.g. "authentication", "refactoring", "bug-fix", "testing")

Focus on WHAT was accomplished, WHY decisions were made, WHAT boundaries/invariants \
were established, and WHAT risks exist.\
"""

# Minimum time gap (seconds) between events to prefer splitting a chunk there
_CHUNK_GAP_THRESHOLD_SECONDS = 300  # 5 minutes


def _parse_event_ts(ev: dict) -> float | None:
    """Parse an event timestamp to epoch seconds."""
    from datetime import datetime as _dt

    ts = ev.get("ts", "")
    if isinstance(ts, str) and ts:
        try:
            return _dt.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    elif isinstance(ts, _dt):
        return ts.timestamp()
    return None


def chunk_events(events: list[dict], max_chars: int = 6000) -> list[list[dict]]:
    """Split events into chunks that fit within the char budget.

    Prefers splitting at natural boundaries (time gaps > 5 minutes between events).
    Falls back to size-based splitting when no natural boundary exists.
    """
    if not events:
        return []

    # Check if everything fits in a single chunk
    test_context = build_prompt_context(events, max_chars=max_chars)
    if "more events)" not in test_context:
        return [events]

    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_chars = 0
    # Estimate ~80 chars per event line on average
    chars_per_event = 80

    for i, ev in enumerate(events):
        est_chars = current_chars + chars_per_event

        if current_chunk and est_chars > max_chars * 0.85:
            # Budget nearly full — try to find a natural boundary nearby
            split_here = True

            # Look ahead: if there's a time gap soon, wait for it
            if est_chars < max_chars:
                prev_ts = _parse_event_ts(events[i - 1]) if i > 0 else None
                curr_ts = _parse_event_ts(ev)
                if prev_ts and curr_ts and (curr_ts - prev_ts) >= _CHUNK_GAP_THRESHOLD_SECONDS:
                    split_here = True
                else:
                    split_here = False

            if split_here:
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0
        elif current_chunk and i > 0:
            # Even if budget isn't full, split at large time gaps
            prev_ts = _parse_event_ts(events[i - 1])
            curr_ts = _parse_event_ts(ev)
            if (
                prev_ts
                and curr_ts
                and (curr_ts - prev_ts) >= _CHUNK_GAP_THRESHOLD_SECONDS
                and current_chars > max_chars * 0.3
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0

        current_chunk.append(ev)
        current_chars += chars_per_event

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def needs_chunked_processing(events: list[dict], max_chars: int = 6000) -> bool:
    """Check if a session has too many events for single-pass summarization."""
    test_context = build_prompt_context(events, max_chars=max_chars)
    return "more events)" in test_context


def build_extraction_prompt(
    session_id: str,
    chunk_index: int,
    total_chunks: int,
    context_text: str,
) -> str:
    """Build the user prompt for chunk fact extraction."""
    return f"""\
Session: {session_id}
Chunk: {chunk_index + 1} of {total_chunks}

Timeline:
{context_text}"""


def build_synthesis_prompt(
    session_id: str,
    project: str,
    agent: str,
    branch: str,
    duration_minutes: int,
    tools: list[str],
    files: list[str],
    merged_facts: dict[str, list[str]],
) -> str:
    """Build the user prompt for final synthesis from merged extracted facts."""
    tool_list = ", ".join(tools) if tools else "none"
    file_list = ", ".join(files) if files else "none"

    facts_text = ""
    for category, items in merged_facts.items():
        if items:
            label = category.replace("_", " ").title()
            facts_text += f"\n{label}:\n"
            for item in items:
                facts_text += f"  - {item}\n"

    return f"""\
Session: {session_id}
Project: {project}
Agent: {agent}
Branch: {branch}
Duration: {duration_minutes} minutes
Tools used: {tool_list}
Files touched: {file_list}

Extracted facts from all session chunks:
{facts_text}"""


EXTRACTION_FACT_KEYS = [
    "files_changed",
    "decisions",
    "constraints",
    "bugs_resolved",
    "tradeoffs",
    "dependencies_changed",
    "errors_encountered",
    "test_actions",
    "security_relevant",
    "rollback_risks",
    "boundaries",
]


def merge_extractions(extractions: list[dict]) -> dict[str, list[str]]:
    """Merge extracted facts from multiple chunks, deduplicating entries."""
    merged: dict[str, list[str]] = {key: [] for key in EXTRACTION_FACT_KEYS}
    seen: dict[str, set[str]] = {key: set() for key in EXTRACTION_FACT_KEYS}

    for extraction in extractions:
        for key in EXTRACTION_FACT_KEYS:
            items = extraction.get(key, [])
            if not isinstance(items, list):
                items = [str(items)] if items else []
            for item in items:
                normalized = str(item).strip()
                if normalized and normalized not in seen[key]:
                    seen[key].add(normalized)
                    merged[key].append(normalized)

    return merged


def build_user_prompt(
    session_id: str,
    project: str,
    agent: str,
    branch: str,
    duration_minutes: int,
    tools: list[str],
    files: list[str],
    context_text: str,
) -> str:
    """Build the user prompt for the LLM."""
    tool_list = ", ".join(tools) if tools else "none"
    file_list = ", ".join(files) if files else "none"
    return f"""\
Session: {session_id}
Project: {project}
Agent: {agent}
Branch: {branch}
Duration: {duration_minutes} minutes
Tools used: {tool_list}
Files touched: {file_list}

Timeline:
{context_text}"""
