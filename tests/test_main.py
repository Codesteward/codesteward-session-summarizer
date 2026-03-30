from unittest.mock import AsyncMock

import pytest

from summarizer.config import Settings
from summarizer.context_builder import (
    EXTRACTION_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from summarizer.hashing import compute_hash
from summarizer.main import PromptInfo, _load_prompts


@pytest.fixture
def settings():
    return Settings(
        clickhouse_url="http://localhost:8123",
        clickhouse_user="default",
        clickhouse_password="",
        clickhouse_database="audit",
    )


@pytest.fixture
def settings_db_prompts():
    return Settings(
        clickhouse_url="http://localhost:8123",
        clickhouse_user="default",
        clickhouse_password="",
        clickhouse_database="audit",
        prompt_source="database",
    )


@pytest.fixture
def settings_eval_enabled():
    return Settings(
        clickhouse_url="http://localhost:8123",
        clickhouse_user="default",
        clickhouse_password="",
        clickhouse_database="audit",
        evaluation_enabled=True,
    )


class TestLoadPrompts:
    @pytest.mark.asyncio
    async def test_code_defaults(self, settings):
        ch = AsyncMock()
        prompts = await _load_prompts(ch, settings)

        assert "extraction" in prompts
        assert "synthesis" in prompts
        assert "summary" in prompts

        assert prompts["extraction"].prompt_id == ""
        assert prompts["extraction"].prompt_text == EXTRACTION_SYSTEM_PROMPT
        assert prompts["extraction"].prompt_hash == compute_hash(EXTRACTION_SYSTEM_PROMPT)

        assert prompts["summary"].prompt_text == SYSTEM_PROMPT
        assert prompts["synthesis"].prompt_text == SYNTHESIS_SYSTEM_PROMPT

        # Should not query DB when prompt_source=code
        ch.get_active_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_database_prompts_loaded(self, settings_db_prompts):
        ch = AsyncMock()
        ch.get_active_prompt.side_effect = lambda role: {
            "extraction": {
                "prompt_id": "ext-v2",
                "prompt_hash": "hash_ext",
                "prompt_text": "Custom extraction prompt",
            },
            "synthesis": {
                "prompt_id": "syn-v2",
                "prompt_hash": "hash_syn",
                "prompt_text": "Custom synthesis prompt",
            },
            "summary": {
                "prompt_id": "sum-v2",
                "prompt_hash": "hash_sum",
                "prompt_text": "Custom summary prompt",
            },
        }.get(role)

        prompts = await _load_prompts(ch, settings_db_prompts)

        assert prompts["extraction"].prompt_id == "ext-v2"
        assert prompts["extraction"].prompt_text == "Custom extraction prompt"
        assert prompts["extraction"].prompt_hash == "hash_ext"

        assert prompts["synthesis"].prompt_id == "syn-v2"
        assert prompts["summary"].prompt_id == "sum-v2"

    @pytest.mark.asyncio
    async def test_database_fallback_to_code_when_no_active(self, settings_db_prompts):
        ch = AsyncMock()
        # Only extraction has an active prompt in DB
        ch.get_active_prompt.side_effect = lambda role: (
            {
                "prompt_id": "ext-v2",
                "prompt_hash": "hash_ext",
                "prompt_text": "Custom extraction prompt",
            }
            if role == "extraction"
            else None
        )

        prompts = await _load_prompts(ch, settings_db_prompts)

        # extraction loaded from DB
        assert prompts["extraction"].prompt_id == "ext-v2"
        # synthesis and summary fell back to code defaults
        assert prompts["synthesis"].prompt_id == ""
        assert prompts["synthesis"].prompt_text == SYNTHESIS_SYSTEM_PROMPT
        assert prompts["summary"].prompt_id == ""
        assert prompts["summary"].prompt_text == SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_database_fallback_on_error(self, settings_db_prompts):
        ch = AsyncMock()
        ch.get_active_prompt.side_effect = Exception("connection refused")

        prompts = await _load_prompts(ch, settings_db_prompts)

        # All roles should fall back to code defaults
        assert prompts["extraction"].prompt_id == ""
        assert prompts["extraction"].prompt_text == EXTRACTION_SYSTEM_PROMPT
        assert prompts["synthesis"].prompt_id == ""
        assert prompts["summary"].prompt_id == ""


class TestPromptInfo:
    def test_dataclass_fields(self):
        info = PromptInfo(
            prompt_id="test-id",
            prompt_text="test prompt",
            prompt_hash="abc123",
        )
        assert info.prompt_id == "test-id"
        assert info.prompt_text == "test prompt"
        assert info.prompt_hash == "abc123"
