from summarizer.context_builder import (
    build_prompt_context,
    build_user_prompt,
    extract_file_path,
    extract_files_from_events,
    extract_tools_from_events,
)


class TestExtractFilePath:
    def test_extracts_file_path_key(self):
        assert extract_file_path('{"file_path": "/src/main.py"}') == "/src/main.py"

    def test_extracts_path_key(self):
        assert extract_file_path('{"path": "/src/utils.ts"}') == "/src/utils.ts"

    def test_extracts_filename_key(self):
        assert extract_file_path('{"filename": "README.md"}') == "README.md"

    def test_extracts_file_key(self):
        assert extract_file_path('{"file": "config.yaml"}') == "config.yaml"

    def test_returns_none_for_empty(self):
        assert extract_file_path("") is None

    def test_returns_none_for_invalid_json(self):
        assert extract_file_path("not json") is None

    def test_returns_none_for_no_path_keys(self):
        assert extract_file_path('{"command": "ls"}') is None

    def test_returns_none_for_non_string_value(self):
        assert extract_file_path('{"file_path": 123}') is None

    def test_priority_order(self):
        # file_path takes priority over path
        result = extract_file_path('{"file_path": "a.py", "path": "b.py"}')
        assert result == "a.py"


class TestExtractFilesFromEvents:
    def test_deduplicates(self):
        events = [
            {"tool_input": '{"file_path": "/a.py"}'},
            {"tool_input": '{"file_path": "/a.py"}'},
            {"tool_input": '{"file_path": "/b.py"}'},
        ]
        assert extract_files_from_events(events) == ["/a.py", "/b.py"]

    def test_preserves_order(self):
        events = [
            {"tool_input": '{"file_path": "/c.py"}'},
            {"tool_input": '{"file_path": "/a.py"}'},
            {"tool_input": '{"file_path": "/b.py"}'},
        ]
        assert extract_files_from_events(events) == ["/c.py", "/a.py", "/b.py"]

    def test_skips_events_without_paths(self):
        events = [
            {"tool_input": '{"command": "ls"}'},
            {"tool_input": '{"file_path": "/a.py"}'},
        ]
        assert extract_files_from_events(events) == ["/a.py"]


class TestExtractToolsFromEvents:
    def test_deduplicates(self):
        events = [
            {"tool_name": "Read"},
            {"tool_name": "Read"},
            {"tool_name": "Write"},
        ]
        assert extract_tools_from_events(events) == ["Read", "Write"]

    def test_skips_empty(self):
        events = [
            {"tool_name": ""},
            {"tool_name": "Read"},
            {},
        ]
        assert extract_tools_from_events(events) == ["Read"]


class TestBuildPromptContext:
    def _make_event(
        self,
        ts="2024-01-01T10:00:00",
        tool_name="",
        direction="assistant",
        tool_input="",
        assistant_text="",
        thinking="",
    ):
        return {
            "ts": ts,
            "tool_name": tool_name,
            "direction": direction,
            "tool_input": tool_input,
            "assistant_text": assistant_text,
            "thinking": thinking,
        }

    def test_basic_tool_event(self):
        events = [self._make_event(tool_name="Read", tool_input='{"file_path": "/app.py"}')]
        result = build_prompt_context(events)
        assert "[2024-01-01T10:00:00] Read → /app.py" in result

    def test_basic_direction_event(self):
        events = [self._make_event(direction="user")]
        result = build_prompt_context(events)
        assert "[2024-01-01T10:00:00] user" in result

    def test_includes_assistant_text(self):
        events = [self._make_event(assistant_text="I'll fix the bug")]
        result = build_prompt_context(events)
        assert "said: I'll fix the bug" in result

    def test_includes_assistant_text_list(self):
        events = [self._make_event(assistant_text=["First message", "Second message"])]
        result = build_prompt_context(events)
        assert "said: First message" in result
        assert "said: Second message" in result

    def test_includes_thinking(self):
        events = [self._make_event(thinking="Let me analyze this")]
        result = build_prompt_context(events)
        assert "thought: Let me analyze this" in result

    def test_truncates_at_budget(self):
        events = [self._make_event(tool_name=f"Tool{i}") for i in range(100)]
        result = build_prompt_context(events, max_chars=200)
        assert "more events)" in result
        assert len(result) < 400  # reasonable upper bound

    def test_strips_newlines_from_snippets(self):
        events = [self._make_event(assistant_text="line1\nline2\nline3")]
        result = build_prompt_context(events)
        assert "\n  said: line1 line2 line3" in result

    def test_strips_secrets(self):
        events = [self._make_event(assistant_text="password=hunter2 is set")]
        result = build_prompt_context(events)
        assert "hunter2" not in result
        assert "password=***" in result


class TestBuildUserPrompt:
    def test_includes_all_fields(self):
        result = build_user_prompt(
            session_id="sess-123",
            project="myproject",
            agent="claude",
            branch="main",
            duration_minutes=42,
            tools=["Read", "Write"],
            files=["app.py", "test.py"],
            context_text="[10:00] Read → app.py",
        )
        assert "Session: sess-123" in result
        assert "Project: myproject" in result
        assert "Agent: claude" in result
        assert "Branch: main" in result
        assert "Duration: 42 minutes" in result
        assert "Read, Write" in result
        assert "app.py, test.py" in result
        assert "[10:00] Read → app.py" in result

    def test_empty_tools_and_files(self):
        result = build_user_prompt(
            session_id="s",
            project="p",
            agent="a",
            branch="b",
            duration_minutes=0,
            tools=[],
            files=[],
            context_text="",
        )
        assert "Tools used: none" in result
        assert "Files touched: none" in result
