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
