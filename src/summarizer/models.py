from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionSummary:
    session_id: str
    revision: int
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


@dataclass
class ChunkExtraction:
    session_id: str
    revision: int
    chunk_index: int
    chunk_start_ts: datetime
    chunk_end_ts: datetime
    event_count: int

    files_changed: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    bugs_resolved: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    dependencies_changed: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    test_actions: list[str] = field(default_factory=list)
    security_relevant: list[str] = field(default_factory=list)
    rollback_risks: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)

    summarizer_model: str = ""
    summarizer_version: str = "v1"
    extracted_at: datetime = field(default_factory=datetime.utcnow)
