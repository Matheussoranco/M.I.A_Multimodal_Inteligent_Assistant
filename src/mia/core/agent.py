"""
Tool-Calling Agent for M.I.A — Multimodal Intelligent Assistant
================================================================

A production-grade agentic loop with:

1. **Native function calling** — for LLMs with a ``chat()`` API that accepts
   ``tools`` (OpenAI, Ollama >= llama3, Anthropic, Groq, Grok, ...).
2. **ReAct text fallback** — for LLMs that only expose plain-text ``query()``.
3. **Persistent conversation memory** — messages survive across turns so the
   assistant remembers what was discussed earlier in the session.
4. **Token / context-window management** — automatically trims old messages
   when the conversation approaches the model's context limit.
5. **Self-reflection** — after tool execution the agent can re-evaluate its
   plan and correct course before answering.
6. **Streaming-ready** — ``run_stream()`` yields tokens as they arrive.
7. **Graceful error recovery** — tool failures, malformed JSON, quota errors
   are all caught and fed back to the LLM so it can retry or inform the user.

The agent picks the best mode automatically and falls back from (1) to (2)
at runtime if the first native call fails.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from .tool_registry import (
    CORE_TOOLS,
    get_tool_descriptions_text,
    get_tool_names,
    validate_tool_args,
)

# ── SOTA modules (lazy-safe: all are optional) ─────────────────────
try:
    from .planner import TaskPlanner  # noqa: F401
except ImportError:  # pragma: no cover
    TaskPlanner = None  # type: ignore[misc,assignment]

try:
    from .persistent_memory import PersistentMemory  # noqa: F401
except ImportError:  # pragma: no cover
    PersistentMemory = None  # type: ignore[misc,assignment]

try:
    from .orchestrator import AgentOrchestrator  # noqa: F401
except ImportError:  # pragma: no cover
    AgentOrchestrator = None  # type: ignore[misc,assignment]

try:
    from .guardrails import GuardrailsManager  # noqa: F401
except ImportError:  # pragma: no cover
    GuardrailsManager = None  # type: ignore[misc,assignment]

try:
    from .benchmarks import BenchmarkSuite  # noqa: F401
except ImportError:  # pragma: no cover
    BenchmarkSuite = None  # type: ignore[misc,assignment]

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
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ═══════════════════════════════════════════════════════════════════════════════
# Conversation memory
# ═══════════════════════════════════════════════════════════════════════════════


class ConversationMemory:
    """Sliding-window conversation memory with token-budget awareness."""

    def __init__(
        self,
        system_prompt: str,
        max_tokens: int = 120_000,
        avg_chars_per_token: float = 3.5,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self._avg = avg_chars_per_token
        self.messages: List[Dict[str, Any]] = []

    def add(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)

    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """Return [system, ...history] trimmed to fit the context window."""
        system = {"role": "system", "content": self.system_prompt}
        budget = self.max_tokens - self._estimate_tokens(self.system_prompt)
        selected: List[Dict[str, Any]] = []
        used = 0
        for msg in reversed(self.messages):
            cost = self._estimate_tokens(str(msg.get("content", ""))) + 10
            if used + cost > budget:
                break
            selected.append(msg)
            used += cost
        selected.reverse()
        return [system] + selected

    def clear(self) -> None:
        self.messages.clear()

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "user")

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self._avg))


# ═══════════════════════════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are M.I.A (Multimodal Intelligent Assistant), an advanced AI agent \
running locally on the user's computer.  You are designed to be genuinely \
helpful, proactive, and capable of executing complex multi-step tasks.

## Core capabilities
- **Tool use** -- you have access to a rich set of tools for file I/O, \
shell commands, web search, desktop automation, messaging, memory, OCR, \
spreadsheets, presentations, and more.
- **Persistent memory** -- you remember the entire conversation within \
this session.  You can also store and retrieve facts in long-term memory.
- **Self-correction** -- if a tool call fails, you analyse the error and \
try an alternative approach before giving up.
- **Proactive reasoning** -- think step-by-step when the task is complex; \
break it into sub-tasks and use tools iteratively.

## Guidelines
1. Be concise, accurate, and helpful.
2. Use tools when the task requires interacting with the system, files, \
web, or external services.  If the task is purely conversational, answer \
directly without tools.
3. When using tools, briefly explain what you are doing so the user can \
follow along.
4. If a tool call fails, inspect the error, try an alternative approach, \
and only report failure if all retries are exhausted.
5. Always prioritise safety -- ask for confirmation before destructive or \
irreversible operations (deleting files, sending messages, etc.).
6. You can chain multiple tool calls to accomplish complex, multi-step tasks.
7. When uncertain, state your uncertainty rather than guessing.
8. Remember context from earlier in the conversation -- do not ask the user \
to repeat information they already provided.
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
5. If a tool fails, analyse the error and try an alternative.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════


class ToolCallingAgent:
    """LLM agent with tool-use, conversation memory, and auto-fallback.

    SOTA capabilities (all optional — degrade gracefully if absent):
    - **Planning** — DAG-based task decomposition for complex goals.
    - **Persistent memory** — cross-session skill library & RAG context.
    - **Multi-agent orchestration** — specialist sub-agents for delegation.
    - **Guardrails** — output validation, PII redaction, tool-call safety.
    - **Benchmarks** — automated agent evaluation framework.
    - **Streaming** — real token-level streaming from the LLM.
    """

    def __init__(
        self,
        llm: Any,
        action_executor: Any,
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        max_steps: int = 20,
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
        max_context_tokens: int = 120_000,
        on_tool_start: Optional[Callable[[str, Dict], None]] = None,
        on_tool_end: Optional[Callable[[str, Optional[str], Optional[Exception]], None]] = None,
        # ── SOTA modules (all optional) ──
        planner: Optional[Any] = None,
        persistent_memory: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        guardrails: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.executor = action_executor
        self.tools = tools or CORE_TOOLS
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._native_tool_support: Optional[bool] = None

        # Conversation memory
        self.memory = ConversationMemory(
            system_prompt=self.system_prompt,
            max_tokens=max_context_tokens,
        )

        # Callbacks for UI integration
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end

        # ── SOTA modules ────────────────────────────────────────────
        self.planner = planner          # TaskPlanner instance
        self.persistent_memory = persistent_memory  # PersistentMemory instance
        self.orchestrator = orchestrator  # AgentOrchestrator instance
        self.guardrails = guardrails    # GuardrailsManager instance

        # Stats
        self.total_tool_calls = 0
        self.total_tokens_used = 0
        self._session_start = time.time()

    # ── Public API ──────────────────────────────────────────────────

    @property
    def supports_native_tools(self) -> bool:
        """Detect whether the LLM exposes a ``chat()`` method."""
        if self._native_tool_support is None:
            self._native_tool_support = hasattr(self.llm, "chat") and callable(
                getattr(self.llm, "chat", None)
            )
        return self._native_tool_support

    @property
    def session_stats(self) -> Dict[str, Any]:
        """Return stats about the current session."""
        return {
            "turns": self.memory.turn_count,
            "tool_calls": self.total_tool_calls,
            "tokens_used": self.total_tokens_used,
            "uptime_seconds": round(time.time() - self._session_start, 1),
        }

    def run(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a user request, invoking tools as needed.

        Execution pipeline (each stage is skipped if its module is ``None``):
        1. Inject persistent-memory RAG context into the prompt.
        2. Check if the request needs a multi-step *plan* → execute the DAG.
        3. Check if the request should be *delegated* to a specialist agent.
        4. Otherwise run the normal native / ReAct loop.
        5. Validate the output through *guardrails*.
        6. Log the interaction to *persistent memory*.
        """
        effective = self._build_effective_message(user_message, context)

        # 1 ── Persistent-memory RAG injection ───────────────────────
        if self.persistent_memory:
            try:
                mem_ctx = self.persistent_memory.get_context_prompt(effective)
                if mem_ctx:
                    effective = f"{mem_ctx}\n\n{effective}"
            except Exception as exc:
                logger.debug("Persistent memory context failed: %s", exc)

        # Record user message in conversation memory
        self.memory.add({"role": "user", "content": effective})

        # 2 ── Planning (DAG decomposition for complex goals) ────────
        if self.planner:
            try:
                if self.planner.should_plan(effective):
                    answer = self._execute_plan(effective)
                    answer = self._apply_guardrails(answer)
                    self.memory.add({"role": "assistant", "content": answer})
                    self._log_interaction(user_message, answer)
                    return answer
            except Exception as exc:
                logger.warning("Planner failed, falling back to direct: %s", exc)

        # 3 ── Multi-agent orchestration ─────────────────────────────
        if self.orchestrator:
            try:
                if self.orchestrator.should_delegate(effective):
                    answer = self._execute_delegation(effective)
                    answer = self._apply_guardrails(answer)
                    self.memory.add({"role": "assistant", "content": answer})
                    self._log_interaction(user_message, answer)
                    return answer
            except Exception as exc:
                logger.warning("Orchestrator failed, falling back: %s", exc)

        # 4 ── Standard agent loop (native / ReAct) ──────────────────
        if self.supports_native_tools:
            answer = self._run_native(effective)
        else:
            answer = self._run_react(effective)

        # 5 ── Guardrails ────────────────────────────────────────────
        answer = self._apply_guardrails(answer)

        # Record assistant answer in conversation memory
        self.memory.add({"role": "assistant", "content": answer})

        # 6 ── Persistent memory log ─────────────────────────────────
        self._log_interaction(user_message, answer)

        return answer

    def run_stream(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Streaming variant — yields text chunks as they arrive.

        If the LLM supports ``stream_chat()`` (returning an iterator), each
        token/chunk is yielded individually.  Otherwise falls back to the
        non-streaming ``run()`` and yields the complete response as one chunk.
        """
        effective = self._build_effective_message(user_message, context)

        # Persistent-memory RAG injection
        if self.persistent_memory:
            try:
                mem_ctx = self.persistent_memory.get_context_prompt(effective)
                if mem_ctx:
                    effective = f"{mem_ctx}\n\n{effective}"
            except Exception:
                pass

        self.memory.add({"role": "user", "content": effective})

        # ── Try real streaming first ────────────────────────────────
        if hasattr(self.llm, "stream_chat") and callable(self.llm.stream_chat):
            try:
                messages = self.memory.get_messages_for_llm()
                chunks: List[str] = []
                for chunk in self.llm.stream_chat(messages, tools=self.tools or None):
                    # chunk can be a string or a ChatResponse fragment
                    text = ""
                    if isinstance(chunk, str):
                        text = chunk
                    elif hasattr(chunk, "content") and chunk.content:
                        text = chunk.content
                    if text:
                        chunks.append(text)
                        yield text
                full_answer = "".join(chunks)
                full_answer = self._apply_guardrails(full_answer)
                self.memory.add({"role": "assistant", "content": full_answer})
                self._log_interaction(user_message, full_answer)
                return
            except Exception as exc:
                logger.debug("stream_chat failed, falling back: %s", exc)

        # ── Fallback: non-streaming run() ───────────────────────────
        if self.supports_native_tools:
            answer = self._run_native(effective)
        else:
            answer = self._run_react(effective)

        answer = self._apply_guardrails(answer)
        yield answer
        self.memory.add({"role": "assistant", "content": answer})
        self._log_interaction(user_message, answer)

    def reset_conversation(self) -> None:
        """Clear conversation history (start a fresh session)."""
        self.memory.clear()

    # ── Native tool-calling loop ────────────────────────────────────

    def _run_native(self, user_message: str) -> str:
        messages = self.memory.get_messages_for_llm()

        for step in range(self.max_steps):
            try:
                response: ChatResponse = self.llm.chat(
                    messages,
                    tools=self.tools if self.tools else None,
                )
            except Exception as exc:
                logger.error("LLM chat failed (step %d): %s", step, exc)
                if step == 0:
                    logger.info("First call failed -- switching to ReAct mode")
                    self._native_tool_support = False
                    return self._run_react(user_message)
                return f"I encountered an error communicating with the language model: {exc}"

            # Track token usage
            if response.usage:
                self.total_tokens_used += sum(response.usage.values())

            if response.has_tool_calls:
                # Record assistant message that requested the tools
                assistant_msg = self._format_assistant_tool_msg(response)
                messages.append(assistant_msg)

                # Execute every requested tool (with retry)
                for tc in response.tool_calls:  # type: ignore[union-attr]
                    result_text = self._execute_tool_with_retry(tc)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        }
                    )
                    self.total_tool_calls += 1
            else:
                # No tool calls -> final answer
                final = response.content or ""

                # Self-reflection: if tools were used and the answer is
                # suspiciously short, ask the LLM to verify
                if self.total_tool_calls > 0 and step > 0 and len(final) < 20:
                    reflection = self._self_reflect(messages, final)
                    if reflection:
                        return reflection
                return final

        return self._summarize_incomplete(messages)

    # ── Self-reflection ─────────────────────────────────────────────

    def _self_reflect(
        self,
        messages: List[Dict[str, Any]],
        initial_answer: str,
    ) -> Optional[str]:
        """Ask the LLM to verify its answer is complete and accurate."""
        try:
            reflection_msg = {
                "role": "user",
                "content": (
                    "Before responding to the user, review your work:\n"
                    "1. Did you fully address the user's request?\n"
                    "2. Is the information accurate based on tool outputs?\n"
                    "3. Is anything missing?\n\n"
                    "If your answer is complete and correct, repeat it exactly.\n"
                    "If it needs correction, provide the corrected version."
                ),
            }
            check_msgs = messages + [
                {"role": "assistant", "content": initial_answer},
                reflection_msg,
            ]
            resp = self.llm.chat(check_msgs, tools=None)
            if resp.content and len(resp.content) > len(initial_answer):
                return resp.content
        except Exception as exc:
            logger.debug("Self-reflection failed: %s", exc)
        return None

    # ── ReAct text-based fallback ───────────────────────────────────

    def _run_react(self, user_message: str) -> str:
        tool_desc = get_tool_descriptions_text()
        system = _REACT_TEMPLATE.format(tool_descriptions=tool_desc)

        # Include recent conversation context for continuity
        context_summary = self._build_react_context()

        history: List[str] = []
        current_input = f"Task: {user_message}"
        if context_summary:
            current_input = f"Previous conversation context:\n{context_summary}\n\n{current_input}"

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
                result = self._execute_tool_with_retry(tc)
                observation = f"Observation: {result}"
                history.append(observation)
                current_input = observation
            else:
                # Neither action nor final answer — treat as direct reply
                content = response.strip()
                if content:
                    return content

        return self._format_react_timeout(history)

    # ── Tool execution with retry & error recovery ────────────────

    def _execute_tool_with_retry(self, tool_call: ToolCall) -> str:
        """Execute a tool with retry logic and lifecycle callbacks."""
        name = tool_call.name
        args = dict(tool_call.arguments)

        if not validate_tool_args(name, args):
            return (
                f"Error: invalid or missing arguments for tool '{name}'. "
                f"Received: {args}"
            )

        # Route high-level "create_document" to concrete actions
        if name == "create_document":
            fmt = args.pop("format", "docx")
            template = args.pop("template", "proposal")
            name = "create_docx" if fmt == "docx" else "create_pdf"
            args["template"] = template

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Executing tool: %s(%s) [attempt %d/%d]",
                name, args, attempt, self.max_retries,
            )

            if self.on_tool_start:
                try:
                    self.on_tool_start(name, args)
                except Exception:
                    pass

            try:
                result = self.executor.execute(name, args)
                text = str(result) if result is not None else "Done."
                # Truncate very long outputs to protect the context window
                if len(text) > 4000:
                    text = text[:4000] + "\n… [truncated]"

                if self.on_tool_end:
                    try:
                        self.on_tool_end(name, text, None)
                    except Exception:
                        pass

                return text

            except Exception as exc:
                last_error = f"Tool '{name}' failed (attempt {attempt}): {exc}"
                logger.warning(last_error)

                if self.on_tool_end:
                    try:
                        self.on_tool_end(name, None, exc)
                    except Exception:
                        pass

                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)  # exponential-ish backoff

        return last_error or f"Tool '{name}' failed after {self.max_retries} attempts."

    # ── Message formatting helpers ──────────────────────────────────

    @staticmethod
    def _format_assistant_tool_msg(response: ChatResponse) -> Dict[str, Any]:
        """Format a ChatResponse with tool_calls into an OpenAI-style message."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
        }
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

    def _build_react_context(self) -> str:
        """Build a summary of recent conversation for the ReAct prompt."""
        recent = self.memory.get_messages_for_llm()[-6:]
        if not recent:
            return ""
        lines = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content and role in ("user", "assistant"):
                # Truncate long messages for context
                preview = content[:300] + "…" if len(content) > 300 else content
                lines.append(f"{role.title()}: {preview}")
        return "\n".join(lines)

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
