"""
Tool-Calling Agent for M.I.A.

Supports two execution modes:

1. **Native function calling** — for LLMs that expose a ``chat()`` method
   accepting ``tools`` (OpenAI, Ollama ≥ llama3, Anthropic, Groq, …).
   The LLM returns structured ``tool_calls``; the agent executes them and
   feeds the results back until the model emits a final text answer.

2. **ReAct text fallback** — for LLMs that only support plain-text
   ``query()``.  The agent constructs a ``Thought → Action → Observation``
   prompt loop and parses the model output with robust regex + JSON
   recovery.

The agent picks the best mode automatically at construction time and
falls back from (1) to (2) at runtime if the first native call fails.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .tool_registry import (
    CORE_TOOLS,
    get_tool_descriptions_text,
    get_tool_names,
    validate_tool_args,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data classes shared between agent and LLMManager
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ToolCall:
    """A single tool / function call emitted by an LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]

    @classmethod
    def create(cls, name: str, arguments: Dict[str, Any]) -> "ToolCall":
        return cls(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=arguments,
        )


@dataclass
class ChatResponse:
    """Structured response from ``LLMManager.chat()``."""

    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str = "stop"  # "stop" | "tool_calls"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ═══════════════════════════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are M.I.A (Multimodal Intelligent Assistant), a capable AI assistant \
running on the user's computer.  You can help with a wide range of tasks \
by using the tools available to you.

Guidelines:
• Be concise and helpful.
• Use tools when the task requires interacting with the system, files, \
web, or external services.
• If a task does not require any tool, just answer directly.
• When using tools, briefly explain what you are doing.
• If a tool call fails, try an alternative approach or inform the user.
• Always prioritise safety — ask for confirmation before destructive operations.
• You can chain multiple tool calls to accomplish complex, multi-step tasks.
"""

_REACT_TEMPLATE = """\
You are M.I.A (Multimodal Intelligent Assistant), a capable AI assistant \
running on the user's computer.

You have access to the following tools:
{tool_descriptions}

To use a tool you MUST follow this EXACT format:

Thought: <your reasoning about what to do>
Action: <tool_name>
Action Input: <JSON object with the tool parameters>

After each tool execution you will receive an Observation with the result.
Continue reasoning from there.

When you have the final answer (or no tool is needed) respond with:

Thought: <your final reasoning>
Final Answer: <your response to the user>

RULES:
1. Action Input MUST be valid JSON (double-quoted strings).
2. Only use tools from the list above.
3. You may use multiple tools in sequence.
4. Always finish with "Final Answer:" when done.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════


class ToolCallingAgent:
    """LLM agent with tool-use capabilities and automatic fallback."""

    def __init__(
        self,
        llm: Any,
        action_executor: Any,
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        max_steps: int = 15,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.executor = action_executor
        self.tools = tools or CORE_TOOLS
        self.max_steps = max_steps
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._native_tool_support: Optional[bool] = None

    # ── Public API ──────────────────────────────────────────────────

    @property
    def supports_native_tools(self) -> bool:
        """Detect whether the LLM exposes a ``chat()`` method."""
        if self._native_tool_support is None:
            self._native_tool_support = hasattr(self.llm, "chat") and callable(
                getattr(self.llm, "chat", None)
            )
        return self._native_tool_support

    def run(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a user request, invoking tools as needed.

        Parameters
        ----------
        user_message:
            The natural-language request from the user.
        context:
            Optional dict with extra context (image descriptions,
            audio transcriptions, …).

        Returns
        -------
        str
            The agent's final textual response.
        """
        effective = self._build_effective_message(user_message, context)

        if self.supports_native_tools:
            return self._run_native(effective)
        return self._run_react(effective)

    # ── Native tool-calling loop ────────────────────────────────────

    def _run_native(self, user_message: str) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for step in range(self.max_steps):
            try:
                response: ChatResponse = self.llm.chat(
                    messages,
                    tools=self.tools if self.tools else None,
                )
            except Exception as exc:
                logger.error("LLM chat failed (step %d): %s", step, exc)
                if step == 0:
                    logger.info("First call failed — switching to ReAct mode")
                    self._native_tool_support = False
                    return self._run_react(user_message)
                return f"I encountered an error communicating with the language model: {exc}"

            if response.has_tool_calls:
                # Record assistant message that requested the tools
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": (
                                    json.dumps(tc.arguments)
                                    if isinstance(tc.arguments, dict)
                                    else str(tc.arguments)
                                ),
                            },
                        }
                        for tc in (response.tool_calls or [])
                    ],
                }
                messages.append(assistant_msg)

                # Execute every requested tool
                for tc in (response.tool_calls or []):
                    result_text = self._execute_tool(tc)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        }
                    )
            else:
                # No tool calls → final answer
                return response.content or ""

        return self._summarize_incomplete(messages)

    # ── ReAct text-based fallback ───────────────────────────────────

    def _run_react(self, user_message: str) -> str:
        tool_desc = get_tool_descriptions_text()
        system = _REACT_TEMPLATE.format(tool_descriptions=tool_desc)

        history: List[str] = []
        current_input = f"Task: {user_message}"

        for step in range(self.max_steps):
            prompt = (
                system
                + "\n\n"
                + "\n".join(history)
                + "\n"
                + current_input
                + "\n"
            )

            try:
                response = self._query_llm_text(prompt)
            except Exception as exc:
                logger.error("LLM query failed (step %d): %s", step, exc)
                return f"I encountered an error: {exc}"

            history.append(response)

            # 1. Final Answer?
            final = self._extract_final_answer(response)
            if final is not None:
                return final

            # 2. Action?
            action = self._parse_react_action(response)
            if action:
                tool_name, tool_args = action
                tc = ToolCall.create(tool_name, tool_args)
                result = self._execute_tool(tc)
                observation = f"Observation: {result}"
                history.append(observation)
                current_input = observation
            else:
                # Neither action nor final answer — treat as direct reply
                content = response.strip()
                if content:
                    return content

        return self._format_react_timeout(history)

    # ── Tool execution with error recovery ──────────────────────────

    def _execute_tool(self, tool_call: ToolCall) -> str:
        name = tool_call.name
        args = dict(tool_call.arguments)

        if not validate_tool_args(name, args):
            return (
                f"Error: invalid or missing arguments for tool '{name}'. "
                f"Received: {args}"
            )

        logger.info("Executing tool: %s(%s)", name, args)

        # Route high-level "create_document" to concrete actions
        if name == "create_document":
            fmt = args.pop("format", "docx")
            template = args.pop("template", "proposal")
            name = "create_docx" if fmt == "docx" else "create_pdf"
            args["template"] = template

        try:
            result = self.executor.execute(name, args)
            text = str(result) if result is not None else "Done."
            # Truncate very long outputs so we don't blow the context window
            if len(text) > 4000:
                text = text[:4000] + "\n… [truncated]"
            return text
        except Exception as exc:
            msg = f"Tool '{name}' failed: {exc}"
            logger.warning(msg)
            return msg

    # ── Internal helpers ────────────────────────────────────────────

    def _build_effective_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        if not context:
            return user_message
        parts: List[str] = []
        for key, value in context.items():
            if value and key != "text":
                parts.append(f"[{key}: {value}]")
        if parts:
            return " ".join(parts) + "\n" + user_message
        return user_message

    def _query_llm_text(self, prompt: str) -> str:
        if hasattr(self.llm, "query") and callable(self.llm.query):
            return str(self.llm.query(prompt) or "")
        if hasattr(self.llm, "query_model") and callable(self.llm.query_model):
            return str(self.llm.query_model(prompt) or "")
        raise RuntimeError("LLM has no usable query method")

    # ── ReAct parsing (with multiple fallback strategies) ───────────

    def _parse_react_action(
        self, response: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        action_match = re.search(r"Action:\s*([\w_]+)", response)
        if not action_match:
            return None

        tool_name = action_match.group(1).strip()
        # Verify it's a known tool
        if tool_name not in get_tool_names():
            logger.warning("LLM hallucinated unknown tool: %s", tool_name)
            return None

        input_match = re.search(
            r"Action Input:\s*(\{.*?\})", response, re.DOTALL
        )
        if input_match:
            args = self._robust_json_parse(input_match.group(1).strip())
            if args is not None:
                return tool_name, args

        # Fallback: no JSON body — call with empty args
        return tool_name, {}

    @staticmethod
    def _robust_json_parse(raw: str) -> Optional[Dict[str, Any]]:
        """Try progressively looser JSON parsing."""
        for attempt_fn in (
            lambda s: json.loads(s),
            lambda s: json.loads(s.replace("'", '"')),
            lambda s: json.loads(
                re.search(r"\{[^{}]*\}", s, re.DOTALL).group()  # type: ignore[union-attr]
            ),
        ):
            try:
                result = attempt_fn(raw)
                if isinstance(result, dict):
                    return result
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_final_answer(response: str) -> Optional[str]:
        match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Remove trailing artefacts
            answer = re.sub(
                r"\n(Thought|Action|Action Input):.*",
                "",
                answer,
                flags=re.DOTALL,
            )
            return answer if answer else None
        return None

    def _summarize_incomplete(self, messages: List[Dict[str, Any]]) -> str:
        tool_results = [
            m["content"] for m in messages if m.get("role") == "tool"
        ]
        if tool_results:
            summary = (
                "I performed several actions but reached the step limit. "
                "Here's what I found:\n\n"
            )
            for i, result in enumerate(tool_results[-3:], 1):
                summary += f"{i}. {result[:500]}\n"
            return summary
        return (
            "I was unable to complete the task within the allowed "
            "number of steps."
        )

    @staticmethod
    def _format_react_timeout(history: List[str]) -> str:
        for entry in reversed(history[-3:]):
            final = ToolCallingAgent._extract_final_answer(entry)
            if final:
                return final
            if "Observation:" in entry:
                obs = entry.split("Observation:")[-1].strip()
                if obs and len(obs) > 20:
                    return f"Here's what I found: {obs}"
        return "I wasn't able to complete the task. Could you try rephrasing?"
