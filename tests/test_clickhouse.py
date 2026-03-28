from datetime import UTC, datetime

import httpx
import pytest
import respx

from summarizer.clickhouse import ClickHouseClient, compute_session_stats
from summarizer.config import Settings
from summarizer.models import ChunkExtraction, SessionSummary


@pytest.fixture
def settings():
    return Settings(
        clickhouse_url="http://localhost:8123",
        clickhouse_user="default",
        clickhouse_password="",
        clickhouse_database="audit",
    )


@pytest.fixture
def ch_client(settings):
    return ClickHouseClient(settings)


class TestGetUnsummarizedSessions:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_session_ids(self, ch_client):
        respx.post("http://localhost:8123").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"session_id": "sess-1"}, {"session_id": "sess-2"}]},
            )
        )
        result = await ch_client.get_unsummarized_sessions(
            lookback_hours=168,
            cooldown_minutes=30,
            batch_size=10,
            summarizer_version="v1",
        )
        assert result == ["sess-1", "sess-2"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_empty_on_no_results(self, ch_client):
        respx.post("http://localhost:8123").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        result = await ch_client.get_unsummarized_sessions(
            lookback_hours=168,
            cooldown_minutes=30,
            batch_size=10,
            summarizer_version="v1",
        )
        assert result == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_passes_correct_params(self, ch_client):
        route = respx.post("http://localhost:8123").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        await ch_client.get_unsummarized_sessions(
            lookback_hours=48,
            cooldown_minutes=15,
            batch_size=5,
            summarizer_version="v2",
        )
        request = route.calls[0].request
        assert "param_lookback_hours" in str(request.url)
        assert "param_batch_size" in str(request.url)


class TestGetSessionEvents:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_events(self, ch_client):
        events = [
            {"ts": "2024-01-01T10:00:00", "tool_name": "Read", "direction": "tool"},
            {"ts": "2024-01-01T10:01:00", "tool_name": "Write", "direction": "tool"},
        ]
        respx.post("http://localhost:8123").mock(
            return_value=httpx.Response(200, json={"data": events})
        )
        result = await ch_client.get_session_events("sess-1")
        assert len(result) == 2
        assert result[0]["tool_name"] == "Read"


class TestWriteSummary:
    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_insert(self, ch_client):
        route = respx.post("http://localhost:8123").mock(return_value=httpx.Response(200))
        summary = SessionSummary(
            session_id="sess-1",
            revision=1,
            project="myproject",
            agent="claude",
            branch="main",
            user="dev",
            team="eng",
            first_ts=datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
            last_ts=datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
            duration_minutes=60,
            turn_count=20,
            tool_call_count=15,
            total_input_tokens=5000,
            total_output_tokens=3000,
            summary="Fixed auth bug",
            key_decisions=["Switched to sessions"],
            files_modified=["auth.py"],
            tools_used=["Read", "Write"],
            tags=["auth", "bug-fix"],
            summarizer_model="phi3:mini",
            summarized_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            summarizer_version="v1",
        )
        await ch_client.write_summary(summary)
        assert route.called
        body = route.calls[0].request.content.decode()
        assert "INSERT INTO session_summaries" in body
        assert "sess-1" in body
        assert "Fixed auth bug" in body

    @pytest.mark.asyncio
    @respx.mock
    async def test_escapes_single_quotes(self, ch_client):
        route = respx.post("http://localhost:8123").mock(return_value=httpx.Response(200))
        summary = SessionSummary(
            session_id="sess-1",
            revision=1,
            project="project",
            agent="agent",
            branch="main",
            user="",
            team="",
            first_ts=datetime(2024, 1, 1, tzinfo=UTC),
            last_ts=datetime(2024, 1, 1, tzinfo=UTC),
            duration_minutes=0,
            turn_count=0,
            tool_call_count=0,
            total_input_tokens=0,
            total_output_tokens=0,
            summary="User's data wasn't loading",
            key_decisions=["Fixed user's query"],
            files_modified=[],
            tools_used=[],
            tags=[],
            summarizer_model="phi3:mini",
            summarized_at=datetime(2024, 1, 1, tzinfo=UTC),
            summarizer_version="v1",
        )
        await ch_client.write_summary(summary)
        body = route.calls[0].request.content.decode()
        assert "\\'" in body


class TestWriteChunkExtraction:
    @pytest.mark.asyncio
    @respx.mock
    async def test_sends_insert(self, ch_client):
        route = respx.post("http://localhost:8123").mock(return_value=httpx.Response(200))
        chunk = ChunkExtraction(
            session_id="sess-1",
            revision=1,
            chunk_index=0,
            chunk_start_ts=datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
            chunk_end_ts=datetime(2024, 1, 1, 10, 30, tzinfo=UTC),
            event_count=15,
            files_changed=["app.py — added endpoint"],
            decisions=["chose async"],
            constraints=["rate limit"],
            bugs_resolved=[],
            tradeoffs=["skipped caching"],
            dependencies_changed=["added httpx"],
            errors_encountered=[],
            test_actions=["added test_app.py"],
            security_relevant=[],
            rollback_risks=["DB migration"],
            boundaries=["never call this API synchronously"],
            summarizer_model="phi3:mini",
            summarizer_version="v1",
            extracted_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        await ch_client.write_chunk_extraction(chunk)
        assert route.called
        body = route.calls[0].request.content.decode()
        assert "INSERT INTO session_chunk_extractions" in body
        assert "sess-1" in body
        assert "chose async" in body
        assert "DB migration" in body


class TestGetNextRevision:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_next_revision(self, ch_client):
        respx.post("http://localhost:8123").mock(
            return_value=httpx.Response(200, json={"data": [{"max_rev": 3}]})
        )
        result = await ch_client.get_next_revision("sess-1")
        assert result == 4

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_1_for_new_session(self, ch_client):
        respx.post("http://localhost:8123").mock(
            return_value=httpx.Response(200, json={"data": [{"max_rev": 0}]})
        )
        result = await ch_client.get_next_revision("sess-new")
        assert result == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_1_for_empty_result(self, ch_client):
        respx.post("http://localhost:8123").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        result = await ch_client.get_next_revision("sess-new")
        assert result == 1


class TestGetChunkExtractions:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_extractions(self, ch_client):
        data = [
            {
                "chunk_index": 0,
                "chunk_start_ts": "2024-01-01T10:00:00",
                "chunk_end_ts": "2024-01-01T10:30:00",
                "event_count": 15,
                "files_changed": ["app.py"],
                "decisions": ["chose async"],
                "constraints": [],
                "bugs_resolved": [],
                "tradeoffs": [],
                "dependencies_changed": [],
                "errors_encountered": [],
                "test_actions": [],
                "security_relevant": [],
                "rollback_risks": [],
                "boundaries": [],
            },
        ]
        respx.post("http://localhost:8123").mock(
            return_value=httpx.Response(200, json={"data": data})
        )
        result = await ch_client.get_chunk_extractions("sess-1", "v1")
        assert len(result) == 1
        assert result[0]["decisions"] == ["chose async"]


class TestComputeSessionStats:
    def test_basic_stats(self):
        events = [
            {
                "ts": "2024-01-01T10:00:00",
                "direction": "assistant",
                "tool_name": "Read",
                "input_tokens": "100",
                "output_tokens": "200",
                "project": "myproject",
                "agent": "claude",
                "branch": "main",
                "user": "dev",
                "team": "eng",
            },
            {
                "ts": "2024-01-01T10:30:00",
                "direction": "assistant",
                "tool_name": "Write",
                "input_tokens": "150",
                "output_tokens": "300",
                "project": "myproject",
                "agent": "claude",
                "branch": "main",
                "user": "dev",
                "team": "eng",
            },
        ]
        stats = compute_session_stats(events)
        assert stats["duration_minutes"] == 30
        assert stats["turn_count"] == 2
        assert stats["tool_call_count"] == 2
        assert stats["total_input_tokens"] == 250
        assert stats["total_output_tokens"] == 500
        assert stats["project"] == "myproject"

    def test_empty_events(self):
        stats = compute_session_stats([])
        assert stats["duration_minutes"] == 0
        assert stats["turn_count"] == 0

    def test_handles_missing_tokens(self):
        events = [
            {
                "ts": "2024-01-01T10:00:00",
                "direction": "assistant",
                "tool_name": "",
                "input_tokens": None,
                "output_tokens": "",
                "project": "p",
                "agent": "a",
                "branch": "b",
                "user": "",
                "team": "",
            },
        ]
        stats = compute_session_stats(events)
        assert stats["total_input_tokens"] == 0
        assert stats["total_output_tokens"] == 0
