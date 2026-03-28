from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionSummary:
    session_id: str
    project: str
    agent: str
    branch: str
    user: str
    team: str

    first_ts: datetime
    last_ts: datetime
    duration_minutes: int

    turn_count: int
    tool_call_count: int
    total_input_tokens: int
    total_output_tokens: int

    summary: str
    key_decisions: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    summarizer_model: str = ""
    summarized_at: datetime = field(default_factory=datetime.utcnow)
    summarizer_version: str = "v1"
