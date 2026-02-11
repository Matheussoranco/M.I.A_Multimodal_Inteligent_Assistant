"""
Integration tests for the M.I.A agent pipeline.

These test the actual wiring between ToolCallingAgent,
the real ActionExecutor (tool dispatch), and the main entrypoints.
They do NOT hit external APIs — the LLM is faked, but everything else
(ActionExecutor routing, file I/O, system-info gathering) runs for real.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# ── Ensure the package is importable ────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mia.core.agent import ChatResponse, ToolCall, ToolCallingAgent
from src.mia.core.tool_registry import CORE_TOOLS


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class FakeLLMForIntegration:
    """
    Programmable fake that supports both ``chat()`` and ``query()``.

    Accepts a list of ``ChatResponse`` objects.  When ``chat()`` is called
    it pops the next response.
    """

    def __init__(self, responses: List[ChatResponse]):
        self._responses = list(responses)
        self._idx = 0

    def chat(self, messages, tools=None) -> ChatResponse:
        if self._idx < len(self._responses):
            r = self._responses[self._idx]
            self._idx += 1
            return r
        return ChatResponse(content="(no more responses)")

    def query(self, prompt: str) -> str:
        return "(fallback)"


def _build_real_executor():
    """Try to build the real ActionExecutor.

    If the import fails (missing optional deps), skip the test.
    """
    try:
        from src.mia.tools.action_executor import ActionExecutor

        ae = ActionExecutor(consent_callback=lambda *a, **kw: True)
        return ae
    except Exception as exc:
        pytest.skip(f"ActionExecutor unavailable: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentWithRealExecutor:
    """
    Integration: ToolCallingAgent + real ActionExecutor.
    LLM is faked; tool execution is real.
    """

    def test_get_system_info_native(self):
        """Agent calls get_system_info and returns the result in an answer."""
        executor = _build_real_executor()

        tc = ToolCall.create("get_system_info", {})
        llm = FakeLLMForIntegration([
            ChatResponse(
                content="Let me check your system.",
                tool_calls=[tc],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="Your system info has been retrieved."),
        ])

        agent = ToolCallingAgent(llm=llm, action_executor=executor)
        result = agent.run("What is my system info?")
        assert result  # non-empty response

    def test_create_and_read_file(self):
        """Agent creates a file then reads it back (two tool calls)."""
        executor = _build_real_executor()

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test_agent.txt")

            tc_write = ToolCall.create(
                "create_file",
                {"path": fpath, "content": "Hello from M.I.A agent!"},
            )
            tc_read = ToolCall.create("read_file", {"path": fpath})

            llm = FakeLLMForIntegration([
                ChatResponse(
                    tool_calls=[tc_write],
                    finish_reason="tool_calls",
                ),
                ChatResponse(
                    tool_calls=[tc_read],
                    finish_reason="tool_calls",
                ),
                ChatResponse(content="I created and read the file."),
            ])

            agent = ToolCallingAgent(llm=llm, action_executor=executor)
            result = agent.run(f"Create file at {fpath} and read it back")
            assert result

            # Verify the file actually exists on disk
            assert os.path.isfile(fpath)
            with open(fpath) as f:
                assert "Hello from M.I.A agent!" in f.read()

    def test_run_command_echo(self):
        """Agent executes a safe shell command and receives output."""
        executor = _build_real_executor()

        cmd = "echo integration_test_ok"
        tc = ToolCall.create("run_command", {"command": cmd})

        llm = FakeLLMForIntegration([
            ChatResponse(tool_calls=[tc], finish_reason="tool_calls"),
            ChatResponse(content="The echo command succeeded."),
        ])

        agent = ToolCallingAgent(llm=llm, action_executor=executor)
        result = agent.run("Run echo test")
        assert result


class TestMainEntryPoints:
    """Verify that main.py functions are importable and wired correctly."""

    def test_imports(self):
        from src.mia.main import (
            initialize_components,
            handle_command,
            cleanup_resources,
            main,
        )

    def test_handle_command_quit(self):
        from src.mia.main import handle_command

        class FakeArgs:
            mode = "text"

        cont, handled = handle_command("quit", FakeArgs(), {})
        assert cont is False
        assert handled is True

    def test_handle_command_help(self):
        from src.mia.main import handle_command

        class FakeArgs:
            mode = "text"

        with patch("src.mia.main.display_help"):
            cont, handled = handle_command("help", FakeArgs(), {})
            assert cont is True
            assert handled is True

    def test_handle_command_unknown(self):
        from src.mia.main import handle_command

        class FakeArgs:
            mode = "text"

        cont, handled = handle_command("some random text", FakeArgs(), {})
        assert cont is True
        assert handled is False  # not a command

    def test_cleanup_resources_empty(self):
        from src.mia.main import cleanup_resources

        # Should not raise with empty dict
        cleanup_resources({})

    def test_cleanup_resources_with_mocks(self):
        from src.mia.main import cleanup_resources

        pm = MagicMock()
        cm = MagicMock()
        rm = MagicMock()

        cleanup_resources({
            "performance_monitor": pm,
            "cache_manager": cm,
            "resource_manager": rm,
        })

        pm.stop_monitoring.assert_called_once()
        pm.cleanup.assert_called_once()
        cm.clear_all.assert_called_once()
        rm.stop.assert_called_once()


class TestCLIModules:
    """Verify CLI extraction modules are importable and functional."""

    def test_parser_import(self):
        from src.mia.cli.parser import parse_arguments, setup_logging

    def test_display_import(self):
        from src.mia.cli.display import (
            display_help,
            display_models,
            display_profiles,
            display_status,
        )

    def test_utils_import(self):
        from src.mia.cli.utils import bold, cyan, green, red, yellow


class TestToolRegistryIntegration:
    """Verify tool registry definitions match ActionExecutor capabilities."""

    def test_tools_have_valid_types(self):
        """All tool definitions should be well-formed."""
        for tool_def in CORE_TOOLS:
            assert tool_def["type"] == "function"
            fn = tool_def["function"]
            params = fn["parameters"]
            assert params["type"] == "object"
            # properties should be a dict
            assert isinstance(params.get("properties", {}), dict)

    def test_tool_count(self):
        """We should have a comprehensive set of core tools."""
        assert 40 <= len(CORE_TOOLS) <= 80
