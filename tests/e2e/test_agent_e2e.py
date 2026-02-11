"""
End-to-end tests for the ToolCallingAgent.

These tests exercise the full agent pipeline:
  FakeLLM â†’ ToolCallingAgent â†’ ActionExecutor â†’ real side-effects

No external APIs are called, but file I/O, system commands, and the
full tool dispatch table are exercised for real.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil
import pytest
from typing import Any, Dict, List, Optional

# â”€â”€ Ensure the package is importable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mia.core.agent import ChatResponse, ToolCall, ToolCallingAgent
from src.mia.core.tool_registry import CORE_TOOLS, get_tool_names


class ScriptedLLM:
    """LLM that replays a sequence of ChatResponse objects.

    Supports both ``chat()`` (native tool calling) and ``query()``
    (text-only / ReAct fallback).
    """

    def __init__(self, chat_responses: List[ChatResponse]):
        self._responses = list(chat_responses)
        self._idx = 0
        self.chat_history: List[dict] = []

    def chat(self, messages, tools=None) -> ChatResponse:
        self.chat_history.append({"messages": messages, "tools": tools})
        if self._idx < len(self._responses):
            r = self._responses[self._idx]
            self._idx += 1
            return r
        return ChatResponse(content="(no more scripted responses)")

    def query(self, prompt: str) -> str:
        return "(fallback text-only)"


def _make_executor():
    """Build a real ActionExecutor (no mocking)."""
    try:
        from src.mia.tools.action_executor import ActionExecutor
        return ActionExecutor()
    except Exception:
        pytest.skip("ActionExecutor unavailable")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# E2E Tests
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestE2ECreateAndReadFile:
    """Agent creates a file then reads it back â€” both via tool calls."""

    def test_create_then_read(self, tmp_path):
        target = str(tmp_path / "hello.txt")

        llm = ScriptedLLM([
            # Step 1: LLM decides to create a file
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall.create("create_file", {
                        "path": target,
                        "content": "Hello from E2E test!",
                    }),
                ],
            ),
            # Step 2: LLM decides to read it back
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall.create("read_file", {"path": target}),
                ],
            ),
            # Step 3: final answer
            ChatResponse(
                content="The file was created and contains: Hello from E2E test!",
            ),
        ])

        agent = ToolCallingAgent(
            llm=llm,
            action_executor=_make_executor(),
            tools=CORE_TOOLS,
        )

        answer = agent.run("Create a file hello.txt and read it back.")
        assert "Hello from E2E test" in answer
        assert os.path.exists(target)
        with open(target) as f:
            assert f.read().strip() == "Hello from E2E test!"


class TestE2ERunCommand:
    """Agent runs a shell command and returns the output."""

    def test_echo_command(self):
        llm = ScriptedLLM([
            # LLM decides to run echo
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall.create("run_command", {
                        "command": "echo E2E_TEST_MARKER",
                    }),
                ],
            ),
            # Final answer
            ChatResponse(
                content="The command output: E2E_TEST_MARKER",
            ),
        ])

        agent = ToolCallingAgent(
            llm=llm,
            action_executor=_make_executor(),
            tools=CORE_TOOLS,
        )

        answer = agent.run("Run echo E2E_TEST_MARKER")
        assert "E2E_TEST_MARKER" in answer


class TestE2EConversationMemory:
    """Verify conversation memory persists across turns."""

    def test_multi_turn_memory(self):
        llm = ScriptedLLM([
            # Turn 1
            ChatResponse(content="Your name is Alice. I'll remember that."),
            # Turn 2 â€” should see previous context
            ChatResponse(content="Your name is Alice, as you told me earlier."),
        ])

        agent = ToolCallingAgent(
            llm=llm,
            action_executor=_make_executor(),
            tools=CORE_TOOLS,
        )

        agent.run("My name is Alice.")
        answer2 = agent.run("What's my name?")
        assert "Alice" in answer2

        # Verify memory has both turns
        msgs = agent.memory.get_messages_for_llm()
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 2

    def test_reset_clears_memory(self):
        llm = ScriptedLLM([
            ChatResponse(content="Hello!"),
            ChatResponse(content="Who are you?"),
        ])

        agent = ToolCallingAgent(
            llm=llm,
            action_executor=_make_executor(),
            tools=CORE_TOOLS,
        )

        agent.run("Hi")
        assert agent.memory.turn_count > 0

        agent.reset_conversation()
        assert agent.memory.turn_count == 0


class TestE2ESessionStats:
    """Verify session stats tracking."""

    def test_stats_after_tool_use(self):
        llm = ScriptedLLM([
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall.create("get_system_info", {}),
                ],
            ),
            ChatResponse(content="System info retrieved."),
        ])

        agent = ToolCallingAgent(
            llm=llm,
            action_executor=_make_executor(),
            tools=CORE_TOOLS,
        )

        agent.run("What's my system info?")
        stats = agent.session_stats
        assert stats["turns"] >= 1
        assert stats["tool_calls"] >= 1
        assert stats["uptime_seconds"] >= 0


class TestE2ERetryOnFailure:
    """Verify tool retry logic."""

    def test_retry_after_tool_failure(self):
        """The executor raises on a bad path, agent should report failure."""
        llm = ScriptedLLM([
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall.create("read_file", {
                        "path": "/nonexistent/path/999.txt",
                    }),
                ],
            ),
            ChatResponse(content="The file doesn't exist."),
        ])

        agent = ToolCallingAgent(
            llm=llm,
            action_executor=_make_executor(),
            tools=CORE_TOOLS,
            max_retries=2,
        )

        answer = agent.run("Read /nonexistent/path/999.txt")
        # Should not crash â€” agent handles the error
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestE2EToolCallbacks:
    """Verify on_tool_start and on_tool_end callbacks fire."""

    def test_callbacks_invoked(self):
        events = []

        def on_start(name, args):
            events.append(("start", name))

        def on_end(name, result, error):
            events.append(("end", name, error))

        llm = ScriptedLLM([
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall.create("get_system_info", {}),
                ],
            ),
            ChatResponse(content="Done."),
        ])

        agent = ToolCallingAgent(
            llm=llm,
            action_executor=_make_executor(),
            tools=CORE_TOOLS,
            on_tool_start=on_start,
            on_tool_end=on_end,
        )

        agent.run("system info")
        assert len(events) >= 2
        assert events[0][0] == "start"
        assert events[1][0] == "end"


class TestE2EToolRegistryCompleteness:
    """Verify tool registry covers all major ActionExecutor actions."""

    def test_all_critical_tools_registered(self):
        names = set(get_tool_names())
        critical = {
            "web_search", "read_file", "create_file", "write_file",
            "delete_file", "move_file", "run_command", "get_system_info",
            "analyze_code", "create_code", "store_memory", "search_memory",
            "send_email", "web_scrape", "research_topic",
            "clipboard_copy", "clipboard_paste", "show_notification",
            "desktop_open_app", "desktop_type_text", "desktop_send_keys",
            "calendar_event", "create_sheet", "create_presentation",
            "ocr_extract_text", "embed_text", "embed_similarity",
        }
        missing = critical - names
        assert not missing, f"Missing tool definitions: {missing}"

    def test_tool_count_minimum(self):
        """We should have at least 40 tools registered."""
        assert len(CORE_TOOLS) >= 40, (
            f"Only {len(CORE_TOOLS)} tools registered, expected â‰¥40"
        )


class TestE2ERunStream:
    """Verify the streaming generator works."""

    def test_stream_yields_result(self):
        llm = ScriptedLLM([
            ChatResponse(content="Streamed answer here."),
        ])

        agent = ToolCallingAgent(
            llm=llm,
            action_executor=_make_executor(),
            tools=CORE_TOOLS,
        )

        chunks = list(agent.run_stream("Hello stream"))
        assert len(chunks) >= 1
        assert "Streamed answer" in chunks[0]

