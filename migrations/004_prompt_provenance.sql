ALTER TABLE audit.session_summaries
    ADD COLUMN IF NOT EXISTS prompt_id String DEFAULT '',
    ADD COLUMN IF NOT EXISTS prompt_hash String DEFAULT '',
    ADD COLUMN IF NOT EXISTS input_context_hash String DEFAULT '';

ALTER TABLE audit.session_chunk_extractions
    ADD COLUMN IF NOT EXISTS prompt_id String DEFAULT '',
    ADD COLUMN IF NOT EXISTS prompt_hash String DEFAULT '',
    ADD COLUMN IF NOT EXISTS input_context_hash String DEFAULT '';
