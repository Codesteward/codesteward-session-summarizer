from __future__ import annotations

import httpx
import structlog
import yaml

logger = structlog.get_logger()


class OllamaClient:
    """Async client for the Ollama HTTP API."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                tags = resp.json()
                available = [m["name"] for m in tags.get("models", [])]
                if self.model in available:
                    logger.info("llm_health_check", model=self.model, status="ready")
                    return True
                # Check without tag suffix (e.g. "phi3:mini" matches "phi3:mini")
                base_name = self.model.split(":")[0]
                for name in available:
                    if name.startswith(base_name):
                        logger.info("llm_health_check", model=self.model, status="ready")
                        return True
                logger.warning(
                    "llm_health_check",
                    model=self.model,
                    status="model_not_found",
                    available=available,
                )
                return False
        except httpx.HTTPError as exc:
            logger.error("llm_health_check", model=self.model, status="unreachable", error=str(exc))
            return False

    async def pull_model(self) -> bool:
        """Pull the configured model from Ollama."""
        logger.info("llm_model_pull", model=self.model, status="starting")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model, "stream": False},
                )
                resp.raise_for_status()
                logger.info("llm_model_pull", model=self.model, status="complete")
                return True
        except httpx.HTTPError as exc:
            logger.error("llm_model_pull", model=self.model, status="failed", error=str(exc))
            return False

    async def ensure_model(self) -> bool:
        """Ensure the model is available, pulling it if necessary."""
        if await self.health_check():
            return True
        logger.info("llm_model_pull", model=self.model, reason="model_not_found")
        return await self.pull_model()

    async def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama generate API and return raw response text."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1024,
                    },
                },
            )
            resp.raise_for_status()
            return resp.json()["response"]

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Call Ollama generate API and parse YAML response.

        Returns parsed dict with keys: summary, key_decisions, tags.
        On parse failure, returns dict with summary set to raw text and empty lists.
        """
        raw = await self._call_api(system_prompt, user_prompt)
        return parse_llm_response(raw)

    async def generate_extraction(self, system_prompt: str, user_prompt: str) -> dict:
        """Call Ollama generate API and parse as extraction response.

        Returns parsed dict with fact category keys, each as a list of strings.
        """
        raw = await self._call_api(system_prompt, user_prompt)
        return parse_extraction_response(raw)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned


def parse_llm_response(raw: str) -> dict:
    """Parse YAML response from the LLM.

    Handles common issues: markdown code fences, invalid YAML, missing fields.
    """
    cleaned = _strip_code_fences(raw)

    try:
        parsed = yaml.safe_load(cleaned)
    except yaml.YAMLError:
        logger.warning("llm_response_parse_failed", raw_length=len(raw))
        return {
            "summary": raw.strip(),
            "key_decisions": [],
            "tags": [],
        }

    if not isinstance(parsed, dict):
        return {
            "summary": raw.strip(),
            "key_decisions": [],
            "tags": [],
        }

    # Normalize fields
    summary = parsed.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)

    key_decisions = parsed.get("key_decisions", [])
    if not isinstance(key_decisions, list):
        key_decisions = [str(key_decisions)]
    key_decisions = [str(d).lstrip("- ") for d in key_decisions if d]

    tags = parsed.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]
    tags = [str(t).strip().lower() for t in tags if t]

    return {
        "summary": summary,
        "key_decisions": key_decisions,
        "tags": tags,
    }


_EXTRACTION_KEYS = [
    "files_changed",
    "decisions",
    "constraints",
    "bugs_resolved",
    "tradeoffs",
    "dependencies_changed",
    "errors_encountered",
    "test_actions",
    "security_relevant",
    "rollback_risks",
    "boundaries",
]


def parse_extraction_response(raw: str) -> dict:
    """Parse YAML extraction response from the LLM.

    Returns dict with all fact category keys, each as a list of strings.
    On parse failure, returns empty lists for all categories.
    """
    cleaned = _strip_code_fences(raw)

    try:
        parsed = yaml.safe_load(cleaned)
    except yaml.YAMLError:
        logger.warning("extraction_response_parse_failed", raw_length=len(raw))
        return {key: [] for key in _EXTRACTION_KEYS}

    if not isinstance(parsed, dict):
        return {key: [] for key in _EXTRACTION_KEYS}

    result: dict[str, list[str]] = {}
    for key in _EXTRACTION_KEYS:
        items = parsed.get(key, [])
        if not isinstance(items, list):
            items = [str(items)] if items else []
        result[key] = [str(item).strip() for item in items if item]

    return result
