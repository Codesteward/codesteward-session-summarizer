-- +goose NO TRANSACTION

-- +goose Up
CREATE TABLE IF NOT EXISTS audit.chunk_evaluation_contexts (
    session_id         String,
    revision           UInt32,
    chunk_index        UInt16,
    prompt_id          String,
    prompt_hash        String,
    input_context_hash String,
    input_context      String,             -- full rendered user prompt text (2-20KB)
    stored_at          DateTime64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(stored_at)
ORDER BY (session_id, revision, chunk_index)
TTL stored_at + INTERVAL 90 DAY;

CREATE TABLE IF NOT EXISTS audit.summary_evaluation_contexts (
    session_id         String,
    revision           UInt32,
    prompt_id          String,
    prompt_hash        String,
    input_context_hash String,
    input_context      String,             -- full rendered user prompt text
    stored_at          DateTime64(3)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(stored_at)
ORDER BY (session_id, revision)
TTL stored_at + INTERVAL 90 DAY;
