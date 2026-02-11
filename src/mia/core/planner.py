"""
Task Planner — DAG-based goal decomposition for M.I.A
=====================================================

Breaks complex user requests into ordered sub-tasks with dependencies,
then executes them respecting the dependency graph.

Key features:
- LLM-powered decomposition of natural-language goals into sub-tasks
- DAG execution order (topological sort) ensuring prerequisites run first
- Parallel-ready: independent sub-tasks are grouped into execution tiers
- Progress tracking with per-task status and results
- Re-planning on failure: if a step fails, the planner asks the LLM to adapt
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────────


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubTask:
    """A single step in a plan."""

    id: str
    title: str
    description: str
    tool_hint: Optional[str] = None  # suggested tool name
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    attempt: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tool_hint": self.tool_hint,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result[:200] if self.result else None,
            "error": self.error,
        }


@dataclass
class Plan:
    """An ordered collection of sub-tasks for achieving a goal."""

    goal: str
    tasks: List[SubTask] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for t in self.tasks
        )

    @property
    def has_failures(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)

    @property
    def summary(self) -> str:
        lines = [f"Plan: {self.goal}"]
        for t in self.tasks:
            icon = {
                TaskStatus.PENDING: "⬜",
                TaskStatus.RUNNING: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.SKIPPED: "⏭️",
            }.get(t.status, "?")
            lines.append(f"  {icon} [{t.id}] {t.title}")
        return "\n".join(lines)

    def get_ready_tasks(self) -> List[SubTask]:
        """Return tasks whose dependencies are all completed."""
        completed_ids = {
            t.id for t in self.tasks
            if t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        }
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.depends_on)
        ]

    def get_execution_tiers(self) -> List[List[SubTask]]:
        """Group tasks into tiers for parallel execution."""
        tiers: List[List[SubTask]] = []
        completed: set = set()
        remaining = [t for t in self.tasks if t.status == TaskStatus.PENDING]

        while remaining:
            tier = [
                t for t in remaining
                if all(d in completed for d in t.depends_on)
            ]
            if not tier:
                # Circular dependency or broken plan — just add everything left
                tiers.append(remaining)
                break
            tiers.append(tier)
            completed.update(t.id for t in tier)
            remaining = [t for t in remaining if t not in tier]

        return tiers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "completed": self.completed_count,
            "total": len(self.tasks),
        }


# ── Planner prompt ──────────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are a task planner. Given a user's goal, decompose it into concrete, \
actionable sub-tasks that can be executed sequentially.

Available tools: {tool_names}

Output a JSON array. Each element must have:
- "id": short unique identifier like "t1", "t2", etc.
- "title": short description (5-10 words)
- "description": detailed instruction for this step
- "tool_hint": the most likely tool name to use (or null if no tool needed)
- "depends_on": array of task IDs that must complete first (empty for first tasks)

Rules:
1. Keep the plan to 2-8 steps — be concise.
2. Each step should be independently verifiable.
3. Only reference tools that actually exist in the available tools list.
4. The dependency graph must be acyclic (no circular dependencies).
5. Simple requests (greeting, Q&A) should have just 1 step with tool_hint null.

Respond with ONLY the JSON array, no markdown fences, no explanation.

User's goal: {goal}
"""

_REPLAN_PROMPT = """\
The following plan step failed:
- Task: {task_title}
- Error: {error}

Original goal: {goal}
Completed so far: {completed}

Suggest a revised approach for this step (1-2 sentences), or say "SKIP" if \
the goal can be achieved without it. Respond with just the revised instruction \
or "SKIP".
"""


# ── TaskPlanner ─────────────────────────────────────────────────────────────


class TaskPlanner:
    """Decomposes complex goals into executable sub-task plans."""

    def __init__(
        self,
        llm_query: Callable[[str], str],
        available_tools: Optional[List[str]] = None,
        max_replan_attempts: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        llm_query:
            Callable that takes a prompt string and returns the LLM's text
            response. This decouples the planner from a specific LLM class.
        available_tools:
            List of tool names the agent can use.
        max_replan_attempts:
            How many times to re-plan a failed step before marking it failed.
        """
        self.llm_query = llm_query
        self.tool_names = available_tools or []
        self.max_replan_attempts = max_replan_attempts

    def create_plan(self, goal: str) -> Plan:
        """Ask the LLM to decompose a goal into sub-tasks."""
        prompt = _PLAN_PROMPT.format(
            tool_names=", ".join(self.tool_names) if self.tool_names else "none",
            goal=goal,
        )

        try:
            raw = self.llm_query(prompt)
            tasks_data = self._parse_plan_json(raw)
        except Exception as exc:
            logger.warning("Plan generation failed: %s — creating single-step plan", exc)
            tasks_data = [
                {
                    "id": "t1",
                    "title": "Execute request directly",
                    "description": goal,
                    "tool_hint": None,
                    "depends_on": [],
                }
            ]

        plan = Plan(goal=goal)
        for td in tasks_data:
            plan.tasks.append(
                SubTask(
                    id=td.get("id", f"t{uuid.uuid4().hex[:4]}"),
                    title=td.get("title", "Untitled step"),
                    description=td.get("description", goal),
                    tool_hint=td.get("tool_hint"),
                    depends_on=td.get("depends_on", []),
                )
            )

        # Validate: remove dangling dependencies
        valid_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            task.depends_on = [d for d in task.depends_on if d in valid_ids]

        logger.info("Created plan with %d tasks for: %s", len(plan.tasks), goal)
        return plan

    def should_plan(self, user_message: str) -> bool:
        """Heuristic: decide whether a message needs multi-step planning.

        Simple greetings/questions don't need a plan. Complex requests
        with multiple actions, research tasks, or creation tasks do.
        """
        msg = user_message.lower().strip()

        # Short messages are likely simple
        if len(msg.split()) <= 5:
            return False

        # Planning indicators
        plan_signals = [
            " and ",  # "search X and then create Y"
            " then ",
            "step by step",
            "create a project",
            "build",
            "research",
            "analyze",
            "compare",
            "set up",
            "install and configure",
            "find and",
            "multiple",
            "several",
        ]
        return any(signal in msg for signal in plan_signals)

    def replan_step(
        self, plan: Plan, failed_task: SubTask, error: str
    ) -> Optional[str]:
        """Ask the LLM for a revised approach when a step fails."""
        completed = ", ".join(
            t.title for t in plan.tasks if t.status == TaskStatus.COMPLETED
        ) or "none"

        prompt = _REPLAN_PROMPT.format(
            task_title=failed_task.title,
            error=error,
            goal=plan.goal,
            completed=completed,
        )

        try:
            response = self.llm_query(prompt).strip()
            if response.upper() == "SKIP":
                return None
            return response
        except Exception:
            return None

    # ── Internal ────────────────────────────────────────────────────

    def _parse_plan_json(self, raw: str) -> List[Dict[str, Any]]:
        """Parse the LLM's JSON plan output with fallbacks."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = cleaned.strip().rstrip("`")

        # Try direct parse
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse plan JSON from LLM response: {raw[:200]}")
