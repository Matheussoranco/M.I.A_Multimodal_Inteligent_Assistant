"""
Multi-Agent Orchestrator for M.I.A
===================================

Provides specialized sub-agents (Researcher, Coder, Analyst, Writer) that
can be delegated to by the main agent based on task type.

Key features:
- Role-based specialisation via tailored system prompts
- Automatic routing based on task classification
- Parallel execution of independent sub-agent tasks
- Result aggregation from multiple specialists
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Agent Roles ─────────────────────────────────────────────────────────────


class AgentRole(Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYST = "analyst"
    WRITER = "writer"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"


_ROLE_SYSTEM_PROMPTS: Dict[AgentRole, str] = {
    AgentRole.RESEARCHER: (
        "You are a research specialist. Your job is to gather accurate, "
        "comprehensive information using web search, scraping, and knowledge "
        "retrieval tools. Always cite sources and verify facts from multiple "
        "angles. Prioritize recency and reliability."
    ),
    AgentRole.CODER: (
        "You are an expert software engineer. Write clean, idiomatic, "
        "well-documented code. Follow best practices for the language. "
        "Include error handling and edge cases. Explain your approach."
    ),
    AgentRole.ANALYST: (
        "You are a data analyst. Examine data carefully, identify patterns, "
        "compute statistics, and present findings in a clear, structured way. "
        "Use tables and numbers to support your conclusions."
    ),
    AgentRole.WRITER: (
        "You are a professional writer. Produce clear, well-structured, "
        "engaging text. Match the tone and style to the audience. "
        "Proofread for grammar, clarity, and flow."
    ),
    AgentRole.EXECUTOR: (
        "You are a task execution specialist. Focus on completing the "
        "requested action precisely using the appropriate tools. "
        "Confirm success and report any issues."
    ),
    AgentRole.REVIEWER: (
        "You are a quality reviewer. Critically evaluate the provided work "
        "for correctness, completeness, and quality. Identify specific "
        "improvements and rate confidence (1-10)."
    ),
}

# Keywords that hint at which specialist to use
_ROLE_KEYWORDS: Dict[AgentRole, List[str]] = {
    AgentRole.RESEARCHER: [
        "search", "find", "research", "look up", "what is", "who is",
        "information about", "learn about", "investigate", "latest",
    ],
    AgentRole.CODER: [
        "code", "write a function", "implement", "debug", "fix the bug",
        "create a script", "programming", "refactor", "algorithm",
        "class", "api", "endpoint",
    ],
    AgentRole.ANALYST: [
        "analyze", "analyse", "statistics", "data", "compare",
        "trend", "calculate", "metrics", "performance", "spreadsheet",
    ],
    AgentRole.WRITER: [
        "write", "draft", "compose", "email", "letter", "essay",
        "article", "blog", "summary", "document", "report",
    ],
    AgentRole.EXECUTOR: [
        "run", "execute", "open", "launch", "install", "create file",
        "delete", "move", "copy", "send", "download",
    ],
}


# ── Sub-Agent ───────────────────────────────────────────────────────────────


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""

    role: AgentRole
    task: str
    output: str
    success: bool = True
    confidence: float = 0.8
    tools_used: List[str] = field(default_factory=list)


class SubAgent:
    """A specialized agent with a role-specific system prompt."""

    def __init__(
        self,
        role: AgentRole,
        run_fn: Callable[[str, Optional[str]], str],
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        role:
            The specialisation of this sub-agent.
        run_fn:
            Callable(task, system_override) -> response.  Typically wraps
            ToolCallingAgent.run() with a custom system prompt.
        system_prompt:
            Override the default role prompt.
        """
        self.role = role
        self.run_fn = run_fn
        self.system_prompt = system_prompt or _ROLE_SYSTEM_PROMPTS.get(
            role, ""
        )

    def execute(self, task: str) -> SubAgentResult:
        """Run the sub-agent on a task."""
        try:
            output = self.run_fn(task, self.system_prompt)
            return SubAgentResult(
                role=self.role,
                task=task,
                output=output,
                success=True,
            )
        except Exception as exc:
            logger.error("SubAgent %s failed: %s", self.role.value, exc)
            return SubAgentResult(
                role=self.role,
                task=task,
                output=f"Error: {exc}",
                success=False,
            )


# ── Orchestrator ────────────────────────────────────────────────────────────


class AgentOrchestrator:
    """Routes tasks to specialized sub-agents and aggregates results."""

    def __init__(
        self,
        run_fn: Callable[[str, Optional[str]], str],
        enable_review: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        run_fn:
            Callable(task, system_override) -> response.
            Wraps the main agent's run() with optional system prompt override.
        enable_review:
            If True, complex multi-agent outputs are reviewed by the
            Reviewer agent before returning.
        """
        self.run_fn = run_fn
        self.enable_review = enable_review
        self.agents: Dict[AgentRole, SubAgent] = {}

        # Create one sub-agent per role
        for role in AgentRole:
            self.agents[role] = SubAgent(role=role, run_fn=run_fn)

    def classify_task(self, task: str) -> AgentRole:
        """Determine which specialist is best suited for a task."""
        task_lower = task.lower()
        scores: Dict[AgentRole, float] = {}

        for role, keywords in _ROLE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[role] = score

        if scores:
            return max(scores, key=scores.get)  # type: ignore[arg-type]

        # Default to executor for action-oriented tasks
        return AgentRole.EXECUTOR

    def delegate(self, task: str, role: Optional[AgentRole] = None) -> str:
        """Delegate a task to the appropriate specialist.

        Parameters
        ----------
        task:
            The task description / user request.
        role:
            Force a specific role. If None, auto-classify.

        Returns
        -------
        str
            The specialist's response, optionally reviewed.
        """
        if role is None:
            role = self.classify_task(task)

        logger.info("Delegating to %s: %s", role.value, task[:80])
        agent = self.agents[role]
        result = agent.execute(task)

        if not result.success:
            return result.output

        # Optional review step for quality
        if self.enable_review and role != AgentRole.REVIEWER:
            return self._review(result)

        return result.output

    def delegate_parallel(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[SubAgentResult]:
        """Execute multiple tasks, each routed to the best specialist.

        Parameters
        ----------
        tasks:
            List of dicts with "task" and optional "role" keys.

        Returns
        -------
        List of SubAgentResult, one per input task.

        Note: In the current implementation these run sequentially.
        A future version could use asyncio or threading for true parallelism.
        """
        results: List[SubAgentResult] = []
        for item in tasks:
            task_text = item.get("task", "")
            role_str = item.get("role")
            role = AgentRole(role_str) if role_str else None

            if role is None:
                role = self.classify_task(task_text)

            agent = self.agents[role]
            result = agent.execute(task_text)
            results.append(result)

        return results

    def should_delegate(self, task: str) -> bool:
        """Heuristic: should this task be handled by a specialist?

        Returns True for tasks that clearly match a specialist's domain
        and are complex enough to benefit from specialisation.
        """
        task_lower = task.lower()

        # Short simple queries don't need delegation
        if len(task_lower.split()) <= 4:
            return False

        # Check if any role has a strong match (>=2 keyword hits)
        for keywords in _ROLE_KEYWORDS.values():
            hits = sum(1 for kw in keywords if kw in task_lower)
            if hits >= 2:
                return True

        return False

    # ── Internal ────────────────────────────────────────────────────

    def _review(self, result: SubAgentResult) -> str:
        """Have the Reviewer agent evaluate and potentially improve output."""
        review_prompt = (
            f"Review this {result.role.value}'s output for the task: "
            f'"{result.task}"\n\n'
            f"Output to review:\n{result.output[:2000]}\n\n"
            "If the output is good, respond with just the original output "
            "(optionally with minor improvements). "
            "If it has significant issues, provide a corrected version."
        )

        try:
            reviewer = self.agents[AgentRole.REVIEWER]
            review_result = reviewer.execute(review_prompt)
            if review_result.success and len(review_result.output) > 20:
                return review_result.output
        except Exception as exc:
            logger.debug("Review failed: %s", exc)

        return result.output
