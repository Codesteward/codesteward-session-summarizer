import httpx
import pytest
import respx

from summarizer.config import Settings
from summarizer.llm import (
    AnthropicClient,
    OllamaClient,
    OpenAIClient,
    create_llm_client,
    parse_extraction_response,
    parse_llm_response,
)


class TestParseLlmResponse:
    def test_valid_yaml(self):
        raw = """\
summary: Fixed authentication bug in login flow
key_decisions:
  - Switched from JWT to session tokens
  - Added rate limiting
tags:
  - authentication
  - bug-fix
  - security"""
        result = parse_llm_response(raw)
        assert result["summary"] == "Fixed authentication bug in login flow"
        assert len(result["key_decisions"]) == 2
        assert "Switched from JWT to session tokens" in result["key_decisions"]
        assert "authentication" in result["tags"]
        assert "bug-fix" in result["tags"]

    def test_yaml_with_code_fences(self):
        raw = """\
```yaml
summary: Refactored database layer
key_decisions:
  - Moved to async queries
tags:
  - refactoring
```"""
        result = parse_llm_response(raw)
        assert result["summary"] == "Refactored database layer"
        assert "Moved to async queries" in result["key_decisions"]

    def test_invalid_yaml_returns_raw_text(self):
        raw = "This is not YAML: [invalid"
        result = parse_llm_response(raw)
        assert result["summary"] == raw.strip()
        assert result["key_decisions"] == []
        assert result["tags"] == []

    def test_non_dict_yaml_returns_raw(self):
        raw = "- item1\n- item2"
        result = parse_llm_response(raw)
        assert result["summary"] == raw.strip()

    def test_missing_fields_default(self):
        raw = "summary: Just a summary"
        result = parse_llm_response(raw)
        assert result["summary"] == "Just a summary"
        assert result["key_decisions"] == []
        assert result["tags"] == []

    def test_non_list_key_decisions(self):
        raw = """\
summary: Test
key_decisions: single decision
tags: single-tag"""
        result = parse_llm_response(raw)
        assert result["key_decisions"] == ["single decision"]
        assert result["tags"] == ["single-tag"]

    def test_strips_bullet_prefixes(self):
        raw = """\
summary: Test
key_decisions:
  - "- Added auth"
  - "- Fixed bug"
tags: []"""
        result = parse_llm_response(raw)
        assert "Added auth" in result["key_decisions"][0]

    def test_tags_lowercased(self):
        raw = """\
summary: Test
key_decisions: []
tags:
  - Authentication
  - BUG-FIX"""
        result = parse_llm_response(raw)
        assert "authentication" in result["tags"]
        assert "bug-fix" in result["tags"]

    def test_empty_string(self):
        result = parse_llm_response("")
        assert result["summary"] == ""
        assert result["key_decisions"] == []

    def test_filters_none_values(self):
        raw = """\
summary: Test
key_decisions:
  - Good decision
  -
  - Another decision
tags:
  - valid
  -"""
        result = parse_llm_response(raw)
        assert len(result["key_decisions"]) == 2
        assert len(result["tags"]) == 1


class TestOllamaClientHealthCheck:
    @pytest.mark.asyncio
    @respx.mock
    async def test_healthy_with_model(self):
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={"models": [{"name": "phi3:mini"}, {"name": "llama3:latest"}]},
            )
        )
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        assert await client.health_check() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_healthy_with_base_name_match(self):
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={"models": [{"name": "phi3:latest"}]},
            )
        )
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        assert await client.health_check() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_model_not_found(self):
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={"models": [{"name": "llama3:latest"}]},
            )
        )
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        assert await client.health_check() is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_unreachable(self):
        respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        assert await client.health_check() is False


class TestOllamaClientGenerate:
    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_parses_yaml(self):
        yaml_response = """\
summary: Implemented feature X
key_decisions:
  - Used async approach
tags:
  - feature"""
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": yaml_response})
        )
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        result = await client.generate("system", "user")
        assert result["summary"] == "Implemented feature X"
        assert "Used async approach" in result["key_decisions"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_handles_http_error(self):
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate("system", "user")


class TestOllamaClientGenerateExtraction:
    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_extraction_parses_facts(self):
        yaml_response = """\
files_changed:
  - app.py — added auth middleware
decisions:
  - chose JWT over sessions
constraints: []
bugs_resolved: []
tradeoffs:
  - skipped refresh tokens for now
dependencies_changed:
  - added pyjwt 2.8
errors_encountered: []
test_actions:
  - added test_auth.py
security_relevant:
  - added token validation on all routes
rollback_risks: []
boundaries:
  - never call auth endpoint without rate limiting"""
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=httpx.Response(200, json={"response": yaml_response})
        )
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        result = await client.generate_extraction("system", "user")
        assert "app.py — added auth middleware" in result["files_changed"]
        assert "chose JWT over sessions" in result["decisions"]
        assert result["constraints"] == []
        assert "skipped refresh tokens for now" in result["tradeoffs"]
        assert "added pyjwt 2.8" in result["dependencies_changed"]
        assert "never call auth endpoint without rate limiting" in result["boundaries"]
        assert "added token validation on all routes" in result["security_relevant"]


class TestParseExtractionResponse:
    def test_valid_yaml(self):
        raw = """\
files_changed:
  - app.py — new endpoint
decisions:
  - used async handler
constraints:
  - rate limit of 100 req/s
bugs_resolved: []
tradeoffs: []
dependencies_changed: []
errors_encountered:
  - import error — fixed by adding package
test_actions: []
security_relevant: []
rollback_risks:
  - added DB migration
boundaries:
  - API must remain backward compatible"""
        result = parse_extraction_response(raw)
        assert len(result["files_changed"]) == 1
        assert len(result["decisions"]) == 1
        assert len(result["constraints"]) == 1
        assert result["bugs_resolved"] == []
        assert len(result["errors_encountered"]) == 1
        assert len(result["rollback_risks"]) == 1
        assert len(result["boundaries"]) == 1

    def test_invalid_yaml_returns_empty(self):
        result = parse_extraction_response("not valid yaml: [broken")
        for key in result:
            assert result[key] == []

    def test_non_dict_returns_empty(self):
        result = parse_extraction_response("- item1\n- item2")
        for key in result:
            assert result[key] == []

    def test_with_code_fences(self):
        raw = """\
```yaml
files_changed:
  - config.py — updated defaults
decisions: []
constraints: []
bugs_resolved: []
tradeoffs: []
dependencies_changed: []
errors_encountered: []
test_actions: []
security_relevant: []
rollback_risks: []
boundaries: []
```"""
        result = parse_extraction_response(raw)
        assert "config.py — updated defaults" in result["files_changed"]

    def test_missing_keys_default_to_empty(self):
        raw = "files_changed:\n  - something.py"
        result = parse_extraction_response(raw)
        assert len(result["files_changed"]) == 1
        assert result["decisions"] == []
        assert result["rollback_risks"] == []
        assert result["boundaries"] == []

    def test_non_list_values_coerced(self):
        raw = """\
files_changed: single file
decisions: one decision
constraints: []
bugs_resolved: []
tradeoffs: []
dependencies_changed: []
errors_encountered: []
test_actions: []
security_relevant: []
rollback_risks: []
boundaries: []"""
        result = parse_extraction_response(raw)
        assert result["files_changed"] == ["single file"]
        assert result["decisions"] == ["one decision"]

    def test_filters_empty_items(self):
        raw = """\
files_changed:
  - good item
  -
  - another item
decisions: []
constraints: []
bugs_resolved: []
tradeoffs: []
dependencies_changed: []
errors_encountered: []
test_actions: []
security_relevant: []
rollback_risks: []
boundaries: []"""
        result = parse_extraction_response(raw)
        assert len(result["files_changed"]) == 2


class TestOllamaClientPullModel:
    @pytest.mark.asyncio
    @respx.mock
    async def test_pull_success(self):
        respx.post("http://localhost:11434/api/pull").mock(
            return_value=httpx.Response(200, json={"status": "success"})
        )
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        assert await client.pull_model() is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_pull_failure(self):
        respx.post("http://localhost:11434/api/pull").mock(
            side_effect=httpx.ConnectError("refused")
        )
        client = OllamaClient("http://localhost:11434", "phi3:mini")
        assert await client.pull_model() is False


class TestCreateLlmClient:
    def test_creates_ollama_client(self):
        settings = Settings(llm_provider="ollama", summarizer_model="phi3:mini")
        client = create_llm_client(settings)
        assert isinstance(client, OllamaClient)
        assert client.model == "phi3:mini"

    def test_creates_openai_client(self):
        settings = Settings(
            llm_provider="openai",
            summarizer_model="gpt-4o-mini",
            openai_api_key="sk-test-key",
        )
        client = create_llm_client(settings)
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4o-mini"

    def test_creates_openai_client_with_base_url(self):
        settings = Settings(
            llm_provider="openai",
            summarizer_model="gpt-4o-mini",
            openai_api_key="sk-test-key",
            openai_base_url="https://custom.endpoint.com/v1",
        )
        client = create_llm_client(settings)
        assert isinstance(client, OpenAIClient)

    def test_creates_anthropic_client(self):
        settings = Settings(
            llm_provider="anthropic",
            summarizer_model="claude-haiku-4-5-20251001",
            anthropic_api_key="sk-ant-test-key",
        )
        client = create_llm_client(settings)
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-haiku-4-5-20251001"

    def test_unknown_provider_raises(self):
        settings = Settings(llm_provider="unknown")
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            create_llm_client(settings)

    def test_openai_missing_key_raises(self):
        settings = Settings(llm_provider="openai", openai_api_key="")
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            create_llm_client(settings)

    def test_anthropic_missing_key_raises(self):
        settings = Settings(llm_provider="anthropic", anthropic_api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
            create_llm_client(settings)


class TestOpenAIClientEnsureModel:
    @pytest.mark.asyncio
    async def test_ensure_model_always_true(self):
        client = OpenAIClient(model="gpt-4o-mini", api_key="sk-test")
        assert await client.ensure_model() is True


class TestAnthropicClientEnsureModel:
    @pytest.mark.asyncio
    async def test_ensure_model_always_true(self):
        client = AnthropicClient(model="claude-haiku-4-5-20251001", api_key="sk-ant-test")
        assert await client.ensure_model() is True
