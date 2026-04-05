-- +goose NO TRANSACTION

-- +goose Up
CREATE TABLE IF NOT EXISTS audit.prompt_registry (
    prompt_id          String,
    prompt_role        LowCardinality(String),  -- 'extraction', 'synthesis', 'summary'
    prompt_hash        String,                  -- SHA-256 of prompt_text
    prompt_text        String,
    status             LowCardinality(String),  -- 'active', 'candidate', 'retired', 'rejected'
    parent_id          String DEFAULT '',        -- lineage: which prompt this was derived from
    change_reason      String DEFAULT '',        -- why this variant was created
    created_at         DateTime64(3),
    created_by         LowCardinality(String) DEFAULT ''  -- 'auto-optimizer' or username
) ENGINE = ReplacingMergeTree(created_at)
ORDER BY (prompt_role, prompt_id);
