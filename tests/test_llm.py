import httpx
import pytest
import respx

from summarizer.llm import OllamaClient, parse_llm_response


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
