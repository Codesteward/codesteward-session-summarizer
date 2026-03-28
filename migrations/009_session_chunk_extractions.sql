-- 009_session_chunk_extractions.sql
-- Structured fact extractions from session chunks.
-- Written by codesteward-session-summarizer during incremental summarization.
-- Read back for merge + synthesis into session_summaries.

CREATE TABLE IF NOT EXISTS audit.session_chunk_extractions (
    session_id         String,
    chunk_index        UInt16,
    chunk_start_ts     DateTime64(3),
    chunk_end_ts       DateTime64(3),
    event_count        UInt32,

    -- Extracted facts
    files_changed        Array(String),   -- file path + what changed
    decisions            Array(String),   -- decision + reasoning
    constraints          Array(String),   -- problems, blockers, limitations
    bugs_resolved        Array(String),   -- bug + resolution
    tradeoffs            Array(String),   -- what was skipped and why
    dependencies_changed Array(String),   -- added/removed/upgraded + why
    errors_encountered   Array(String),   -- errors hit and resolution status
    test_actions         Array(String),   -- tests added/modified/skipped
    security_relevant    Array(String),   -- auth, permissions, validation changes
    rollback_risks       Array(String),   -- migrations, schema, irreversible changes
    boundaries           Array(String),   -- explicit rules, invariants, must/must-not constraints

    -- Metadata
    summarizer_model   LowCardinality(String),
    summarizer_version LowCardinality(String),
    extracted_at       DateTime64(3)
) ENGINE = ReplacingMergeTree(extracted_at)
PARTITION BY toYYYYMM(chunk_start_ts)
ORDER BY (session_id, chunk_index);
