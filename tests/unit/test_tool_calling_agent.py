"""
Unit tests for ToolCallingAgent — covers both native and ReAct modes.

Uses lightweight fakes for LLM and ActionExecutor so these tests run
without network, GPU, or any external service.
"""

from __future__ import annotations

import json
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import sys, os

# ── Ensure the package is importable ────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mia.core.agent import (
    ChatResponse,
    ToolCall,
    ToolCallingAgent,
    SYSTEM_PROMPT,
)
from src.mia.core.tool_registry import CORE_TOOLS, get_tool_names


# ═══════════════════════════════════════════════════════════════════════════════
# Fakes
# ═══════════════════════════════════════════════════════════════════════════════


class FakeLLM:
    """Mimics a provider that supports both ``chat()`` and ``query()``."""

    def __init__(
        self,
        chat_responses: Optional[List[ChatResponse]] = None,
        query_responses: Optional[List[str]] = None,
    ):
        self._chat_idx = 0
        self._query_idx = 0
        self._chat_responses = chat_responses or []
        self._query_responses = query_responses or []
        self.chat_calls: List[dict] = []
        self.query_calls: List[str] = []

    def chat(
        self,
        messages: list,
        tools: Optional[list] = None,
    ) -> ChatResponse:
        self.chat_calls.append({"messages": messages, "tools": tools})
        if self._chat_idx < len(self._chat_responses):
            resp = self._chat_responses[self._chat_idx]
            self._chat_idx += 1
            return resp
        return ChatResponse(content="Default answer")

    def query(self, prompt: str) -> str:
        self.query_calls.append(prompt)
        if self._query_idx < len(self._query_responses):
            resp = self._query_responses[self._query_idx]
            self._query_idx += 1
            return resp
        return "Default query answer"


class FakeLLMTextOnly:
    """Mimics a provider that has ``query()`` but NO ``chat()``."""

    def __init__(self, responses: Optional[List[str]] = None):
        self._idx = 0
        self._responses = responses or []
        self.query_calls: List[str] = []

    def query(self, prompt: str) -> str:
        self.query_calls.append(prompt)
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return "Fallback answer"


class FakeActionExecutor:
    """Records tool calls and returns canned results."""

    def __init__(self, results: Optional[Dict[str, str]] = None):
        self.results = results or {}
        self.calls: List[dict] = []

    def execute(self, action: str, params: dict) -> str:
        self.calls.append({"action": action, "params": params})
        if action in self.results:
            return self.results[action]
        return f"Executed {action} successfully"


# ═══════════════════════════════════════════════════════════════════════════════
# ToolCall / ChatResponse tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolCall:
    def test_create_has_id(self):
        tc = ToolCall.create("web_search", {"query": "hello"})
        assert tc.id.startswith("call_")
        assert tc.name == "web_search"
        assert tc.arguments == {"query": "hello"}

    def test_unique_ids(self):
        ids = {ToolCall.create("x", {}).id for _ in range(100)}
        assert len(ids) == 100

    def test_chat_response_no_tool_calls(self):
        r = ChatResponse(content="hi")
        assert not r.has_tool_calls
        assert r.content == "hi"

    def test_chat_response_with_tool_calls(self):
        tc = ToolCall.create("run_command", {"command": "ls"})
        r = ChatResponse(tool_calls=[tc], finish_reason="tool_calls")
        assert r.has_tool_calls


# ═══════════════════════════════════════════════════════════════════════════════
# ToolCallingAgent — native mode
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentNativeMode:
    def test_simple_text_response(self):
        """LLM returns text immediately, no tools used."""
        llm = FakeLLM(chat_responses=[
            ChatResponse(content="The capital of France is Paris."),
        ])
        executor = FakeActionExecutor()
        agent = ToolCallingAgent(llm=llm, action_executor=executor)

        result = agent.run("What is the capital of France?")
        assert "Paris" in result
        assert len(executor.calls) == 0
        assert len(llm.chat_calls) == 1

    def test_supports_native_tools_detection(self):
        llm = FakeLLM()
        agent = ToolCallingAgent(llm=llm, action_executor=FakeActionExecutor())
        assert agent.supports_native_tools is True

        text_llm = FakeLLMTextOnly()
        agent2 = ToolCallingAgent(llm=text_llm, action_executor=FakeActionExecutor())
        assert agent2.supports_native_tools is False

    def test_single_tool_call(self):
        """LLM requests one tool, gets result, then gives final answer."""
        tc = ToolCall.create("web_search", {"query": "python 3.13"})
        llm = FakeLLM(chat_responses=[
            ChatResponse(
                content="Let me search for that.",
                tool_calls=[tc],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="Python 3.13 was released in October 2024."),
        ])
        executor = FakeActionExecutor(results={"web_search": "Python 3.13 released Oct 2024"})
        agent = ToolCallingAgent(llm=llm, action_executor=executor)

        result = agent.run("When was Python 3.13 released?")
        assert "2024" in result
        assert len(executor.calls) == 1
        assert executor.calls[0]["action"] == "web_search"

    def test_multi_step_tool_calls(self):
        """Agent chains two tool calls before final answer."""
        tc1 = ToolCall.create("web_search", {"query": "weather"})
        tc2 = ToolCall.create("get_system_info", {})
        llm = FakeLLM(chat_responses=[
            ChatResponse(
                tool_calls=[tc1],
                finish_reason="tool_calls",
            ),
            ChatResponse(
                tool_calls=[tc2],
                finish_reason="tool_calls",
            ),
            ChatResponse(content="It's sunny and your system is healthy."),
        ])
        executor = FakeActionExecutor(results={
            "web_search": "Sunny, 25C",
            "get_system_info": "CPU: 30%, RAM: 4GB free",
        })
        agent = ToolCallingAgent(llm=llm, action_executor=executor)

        result = agent.run("Weather and system status please")
        assert "sunny" in result.lower()
        assert len(executor.calls) == 2

    def test_tool_execution_error_recovery(self):
        """Agent handles a tool that throws an exception."""
        tc = ToolCall.create("run_command", {"command": "badcmd"})
        llm = FakeLLM(chat_responses=[
            ChatResponse(tool_calls=[tc], finish_reason="tool_calls"),
            ChatResponse(content="The command failed. Try a different approach."),
        ])
        executor = FakeActionExecutor()
        executor.execute = MagicMock(side_effect=RuntimeError("command not found"))

        agent = ToolCallingAgent(llm=llm, action_executor=executor)
        result = agent.run("Run badcmd")
        # Should not crash; should contain the error context or a fallback
        assert result  # got some response

    def test_max_steps_limit(self):
        """Agent stops after max_steps and summarises."""
        # Return tool calls indefinitely
        tc = ToolCall.create("web_search", {"query": "loop"})
        llm = FakeLLM(chat_responses=[
            ChatResponse(tool_calls=[tc], finish_reason="tool_calls"),
        ] * 20)
        executor = FakeActionExecutor()
        agent = ToolCallingAgent(llm=llm, action_executor=executor, max_steps=3)

        result = agent.run("infinite loop test")
        # Should terminate gracefully within 3 steps
        assert len(executor.calls) <= 3
        assert result

    def test_context_is_prepended(self):
        """Extra context (image, audio) is prepended to the user message."""
        llm = FakeLLM(chat_responses=[
            ChatResponse(content="I see a cat."),
        ])
        agent = ToolCallingAgent(llm=llm, action_executor=FakeActionExecutor())

        agent.run("What's in this image?", context={"image": "A tabby cat"})
        sent_msg = llm.chat_calls[0]["messages"][-1]["content"]
        assert "tabby cat" in sent_msg.lower()

    def test_fallback_to_react_on_first_chat_failure(self):
        """If the first chat() call throws, agent falls back to ReAct."""
        llm = FakeLLM()
        llm.chat = MagicMock(side_effect=RuntimeError("API error"))

        # The query() will be used by ReAct fallback
        llm.query = MagicMock(return_value=(
            "Thought: I should answer directly.\n"
            "Final Answer: The answer is 42."
        ))

        executor = FakeActionExecutor()
        agent = ToolCallingAgent(llm=llm, action_executor=executor)

        result = agent.run("What is 6*7?")
        assert "42" in result
        # Should have switched off native tools
        assert agent.supports_native_tools is False


# ═══════════════════════════════════════════════════════════════════════════════
# ToolCallingAgent — ReAct mode
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentReActMode:
    def test_direct_answer_no_tools(self):
        """LLM answers with Final Answer immediately."""
        llm = FakeLLMTextOnly(responses=[
            "Thought: I can answer directly.\nFinal Answer: Hello there!"
        ])
        executor = FakeActionExecutor()
        agent = ToolCallingAgent(llm=llm, action_executor=executor)

        result = agent.run("Say hello")
        assert "Hello there!" in result
        assert len(executor.calls) == 0

    def test_react_with_tool_usage(self):
        """ReAct loop: LLM calls a tool, gets observation, then answers."""
        llm = FakeLLMTextOnly(responses=[
            (
                'Thought: I need to search the web.\n'
                'Action: web_search\n'
                'Action Input: {"query": "capital of Brazil"}'
            ),
            (
                "Thought: I now know the answer.\n"
                "Final Answer: The capital of Brazil is Brasília."
            ),
        ])
        executor = FakeActionExecutor(results={"web_search": "Brasília is the capital of Brazil."})
        agent = ToolCallingAgent(llm=llm, action_executor=executor)

        result = agent.run("What is the capital of Brazil?")
        assert "Brasília" in result
        assert len(executor.calls) == 1

    def test_react_hallucinates_unknown_tool(self):
        """ReAct mode ignores hallucinated tool names."""
        llm = FakeLLMTextOnly(responses=[
            (
                "Thought: I should use a magic tool.\n"
                "Action: magic_wand\n"
                'Action Input: {"spell": "lumos"}'
            ),
            "Thought: That didn't work.\nFinal Answer: I don't have a magic wand.",
        ])
        executor = FakeActionExecutor()
        agent = ToolCallingAgent(llm=llm, action_executor=executor)

        result = agent.run("Cast a spell")
        # The hallucinated tool should NOT have been executed
        assert len(executor.calls) == 0
        assert result

    def test_react_invalid_json_recovery(self):
        """ReAct recovers from malformed JSON in Action Input."""
        llm = FakeLLMTextOnly(responses=[
            (
                "Thought: Let me search.\n"
                "Action: web_search\n"
                "Action Input: {'query': 'test'}"  # single quotes!
            ),
            "Thought: Got it.\nFinal Answer: Test result.",
        ])
        executor = FakeActionExecutor(results={"web_search": "test result"})
        agent = ToolCallingAgent(llm=llm, action_executor=executor)

        result = agent.run("Search for test")
        # Should have recovered via single-quote replacement
        assert len(executor.calls) == 1
        assert "test" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    def test_core_tools_non_empty(self):
        assert len(CORE_TOOLS) > 0

    def test_core_tools_schema(self):
        """Every tool definition follows OpenAI function-calling schema."""
        for tool_def in CORE_TOOLS:
            assert tool_def["type"] == "function"
            fn = tool_def["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"

    def test_get_tool_names_matches(self):
        names = get_tool_names()
        for tool_def in CORE_TOOLS:
            assert tool_def["function"]["name"] in names

    def test_essential_tools_exist(self):
        """The minimum set of useful tools is registered."""
        names = get_tool_names()
        for expected in ["web_search", "run_command", "create_file", "read_file"]:
            assert expected in names, f"Expected tool '{expected}' not found"


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestInternalHelpers:
    def _make_agent(self):
        return ToolCallingAgent(
            llm=FakeLLM(),
            action_executor=FakeActionExecutor(),
        )

    def test_extract_final_answer(self):
        agent = self._make_agent()
        assert agent._extract_final_answer("Final Answer: hello") == "hello"
        assert agent._extract_final_answer("No answer here") is None
        result = agent._extract_final_answer(
            "Thought: ok\nFinal Answer: multi\nline answer\n"
        )
        assert result is not None and result.startswith("multi")

    def test_robust_json_parse(self):
        agent = self._make_agent()
        assert agent._robust_json_parse('{"a": 1}') == {"a": 1}
        assert agent._robust_json_parse("{'a': 1}") == {"a": 1}
        assert agent._robust_json_parse("garbage {\"a\": 1} more") == {"a": 1}
        assert agent._robust_json_parse("totally broken") is None

    def test_build_effective_message_no_context(self):
        agent = self._make_agent()
        assert agent._build_effective_message("hi", None) == "hi"
        assert agent._build_effective_message("hi", {}) == "hi"

    def test_build_effective_message_with_context(self):
        agent = self._make_agent()
        msg = agent._build_effective_message(
            "describe", {"image": "a dog", "audio": "bark"}
        )
        assert "dog" in msg
        assert "bark" in msg
        assert "describe" in msg
