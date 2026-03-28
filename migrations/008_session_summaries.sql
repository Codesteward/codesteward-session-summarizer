-- 008_session_summaries.sql
-- Pre-computed LLM summaries of development sessions.
-- Written by codesteward-session-summarizer, read by codesteward-mcp.

CREATE TABLE IF NOT EXISTS audit.session_summaries (
    session_id        String,
    project           String,
    agent             LowCardinality(String),
    branch            LowCardinality(String),
    user              LowCardinality(String)  DEFAULT '',
    team              LowCardinality(String)  DEFAULT '',

    -- Time boundaries
    first_ts          DateTime64(3),
    last_ts           DateTime64(3),
    duration_minutes  UInt32,

    -- Quantitative stats
    turn_count        UInt32,
    tool_call_count   UInt32,
    total_input_tokens   UInt64,
    total_output_tokens  UInt64,

    -- LLM-generated summaries
    summary           String,          -- 2-4 sentence natural language summary
    key_decisions     Array(String),   -- Bullet points of key decisions/changes
    files_modified    Array(String),   -- Deduplicated file paths
    tools_used        Array(String),   -- Deduplicated tool names
    tags              Array(String),   -- Auto-generated topic tags

    -- Metadata
    summarizer_model  LowCardinality(String),  -- e.g. "phi3:mini", "llama-3.1-8b"
    summarized_at     DateTime64(3),
    summarizer_version LowCardinality(String)   -- Tracks prompt/code version
) ENGINE = ReplacingMergeTree(summarized_at)
PARTITION BY toYYYYMM(first_ts)
ORDER BY (project, session_id);
