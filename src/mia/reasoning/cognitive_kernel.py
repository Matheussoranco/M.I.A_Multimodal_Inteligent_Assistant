"""
Cognitive Kernel for M.I.A
===========================

The *Cognitive Kernel* is the central coordination layer that sits
between the agent loop (``core.agent``) and all intelligence /
reasoning subsystems.  It is responsible for:

1. **Pre-processing** — classifying incoming requests and extracting
   structured features before the agent loop starts.
2. **Strategy selection** — choosing the right mix of algorithmic,
   intelligence, and LLM-based reasoning for each request.
3. **Working memory** — maintaining a short-term scratch-pad for
   intermediate results during multi-step tasks.
4. **Introspection** — monitoring its own performance, detecting
   when it is stuck, and switching strategies.
5. **Skill library** — caching previously synthesised programs and
   solutions for rapid reuse (like human "chunking").

The kernel is designed to be **composable**: every subsystem is
optional and loaded lazily.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Working Memory  (short-term scratchpad)
# ═══════════════════════════════════════════════════════════════════════════


class WorkingMemory:
    """Fixed-capacity key-value store with LRU eviction.

    Models the limited capacity of human working memory.
    Items decay after *ttl* seconds if not refreshed.
    """

    def __init__(self, capacity: int = 32, ttl: float = 300.0) -> None:
        self.capacity = capacity
        self.ttl = ttl
        self._store: OrderedDict[str, Tuple[Any, float]] = OrderedDict()

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time())
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def keys(self) -> List[str]:
        self._evict_expired()
        return list(self._store.keys())

    def clear(self) -> None:
        self._store.clear()

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            del self._store[k]

    def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of current contents."""
        self._evict_expired()
        return {k: v for k, (v, _) in self._store.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Skill Library  (long-term programme / solution cache)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Skill:
    """A reusable, previously-verified solution or programme."""
    id: str
    description: str
    solution: Any  # could be a programme, a function, a template, etc.
    domain: str
    success_count: int = 0
    fail_count: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def reliability(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.5


class SkillLibrary:
    """Persistent store for learned skills (like human "chunking")."""

    def __init__(self, max_skills: int = 500) -> None:
        self.max_skills = max_skills
        self._skills: Dict[str, Skill] = {}

    def store(self, description: str, solution: Any, domain: str = "general") -> str:
        skill_id = hashlib.md5(description.encode()).hexdigest()[:12]
        if skill_id in self._skills:
            self._skills[skill_id].success_count += 1
            return skill_id
        self._skills[skill_id] = Skill(
            id=skill_id, description=description,
            solution=solution, domain=domain,
            success_count=1,
        )
        if len(self._skills) > self.max_skills:
            # Evict least reliable
            worst = min(self._skills.values(), key=lambda s: s.reliability)
            del self._skills[worst.id]
        return skill_id

    def lookup(self, description: str) -> Optional[Skill]:
        """Exact match lookup."""
        skill_id = hashlib.md5(description.encode()).hexdigest()[:12]
        return self._skills.get(skill_id)

    def search(self, query: str, top_k: int = 5) -> List[Skill]:
        """Simple keyword search across skill descriptions."""
        query_words = set(query.lower().split())
        scored: List[Tuple[float, Skill]] = []
        for skill in self._skills.values():
            desc_words = set(skill.description.lower().split())
            overlap = len(query_words & desc_words) / max(len(query_words | desc_words), 1)
            if overlap > 0.1:
                scored.append((overlap * skill.reliability, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def record_outcome(self, skill_id: str, success: bool) -> None:
        if skill_id in self._skills:
            if success:
                self._skills[skill_id].success_count += 1
            else:
                self._skills[skill_id].fail_count += 1

    @property
    def size(self) -> int:
        return len(self._skills)


# ═══════════════════════════════════════════════════════════════════════════
# Introspection Monitor
# ═══════════════════════════════════════════════════════════════════════════


class ExecutionState(Enum):
    IDLE = auto()
    CLASSIFYING = auto()
    REASONING = auto()
    EXECUTING_TOOL = auto()
    REFLECTING = auto()
    STUCK = auto()


@dataclass
class IntrospectionState:
    """Tracks the kernel's internal state for self-monitoring."""
    current_state: ExecutionState = ExecutionState.IDLE
    steps_taken: int = 0
    strategies_tried: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    time_started: float = 0.0
    last_output_hash: str = ""
    repeated_output_count: int = 0

    def is_stuck(self, max_repeats: int = 3, max_time: float = 60.0) -> bool:
        """Detect if the kernel is stuck (repeated output or timeout)."""
        if self.repeated_output_count >= max_repeats:
            return True
        if self.time_started > 0 and (time.time() - self.time_started) > max_time:
            return True
        return False

    def record_output(self, output: str) -> None:
        h = hashlib.md5(output.encode()).hexdigest()
        if h == self.last_output_hash:
            self.repeated_output_count += 1
        else:
            self.repeated_output_count = 0
            self.last_output_hash = h


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Kernel
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveKernel:
    """Central intelligence coordinator.

    Wires together:
    - ``HybridReasoningEngine`` — algorithmic + LLM reasoning
    - ``WorkingMemory`` — short-term scratchpad
    - ``SkillLibrary`` — long-term solution cache
    - ``IntrospectionState`` — self-monitoring

    The kernel pre-processes every request, selects the best strategy,
    and post-processes results before they reach the agent.
    """

    def __init__(
        self,
        llm: Any = None,
        *,
        working_memory_capacity: int = 32,
        skill_library_size: int = 500,
    ) -> None:
        self.llm = llm
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.skill_library = SkillLibrary(max_skills=skill_library_size)
        self.introspection = IntrospectionState()

        # Lazy-loaded reasoning engine
        self._reasoning_engine: Any = None

    @property
    def reasoning(self):
        """Lazy-load the hybrid reasoning engine."""
        if self._reasoning_engine is None:
            from .hybrid_engine import HybridReasoningEngine
            self._reasoning_engine = HybridReasoningEngine(llm=self.llm)
        return self._reasoning_engine

    # ── Main processing pipeline ────────────────────────────────────

    def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full cognitive processing pipeline.

        Returns a dict with:
        - ``answer`` — the result
        - ``confidence`` — 0.0–1.0
        - ``method`` — reasoning method used
        - ``trace`` — step-by-step reasoning log
        - ``from_cache`` — whether a cached skill was used
        """
        self.introspection.current_state = ExecutionState.CLASSIFYING
        self.introspection.time_started = time.time()
        self.introspection.steps_taken = 0

        # 1 ── Check skill library for cached solution ───────────────
        cached = self.skill_library.lookup(query)
        if cached and cached.reliability >= 0.8:
            logger.info("Skill cache hit: %s (reliability=%.2f)", cached.id, cached.reliability)
            return {
                "answer": cached.solution,
                "confidence": cached.reliability,
                "method": "skill_cache",
                "trace": [f"Reused cached skill: {cached.description}"],
                "from_cache": True,
            }

        # 2 ── Store query context in working memory ─────────────────
        self.working_memory.put("current_query", query)
        if context:
            self.working_memory.put("current_context", context)

        # 3 ── Reason ────────────────────────────────────────────────
        self.introspection.current_state = ExecutionState.REASONING

        result = self.reasoning.reason(query, context=context)
        self.introspection.steps_taken += 1

        # 4 ── Introspection: is the answer good enough? ─────────────
        if result.confidence < 0.5 and self.llm is not None:
            self.introspection.current_state = ExecutionState.REFLECTING
            # Try ensemble
            ensemble_result = self.reasoning.reason_ensemble(query, context=context)
            if ensemble_result.confidence > result.confidence:
                result = ensemble_result
                self.introspection.steps_taken += 1

        # 5 ── Check if stuck ────────────────────────────────────────
        self.introspection.record_output(str(result.answer))
        if self.introspection.is_stuck():
            self.introspection.current_state = ExecutionState.STUCK
            logger.warning("Cognitive kernel detected stuck state")
            # Try a completely different approach
            from .hybrid_engine import TaskDomain
            alternative_domains = [
                TaskDomain.NATURAL_LANGUAGE,
                TaskDomain.SEQUENCE_PATTERN,
                TaskDomain.ANALOGY,
            ]
            for alt_domain in alternative_domains:
                if alt_domain != result.domain:
                    alt = self.reasoning.reason(
                        query, context=context, domain_hint=alt_domain,
                    )
                    if alt.confidence > result.confidence:
                        result = alt
                        break

        # 6 ── Cache successful results ──────────────────────────────
        if result.confidence >= 0.7 and result.answer is not None:
            skill_id = self.skill_library.store(
                query, result.answer, domain=result.domain.name,
            )
            self.working_memory.put("last_skill_id", skill_id)

        # 7 ── Store result in working memory ────────────────────────
        self.working_memory.put("last_answer", result.answer)
        self.working_memory.put("last_confidence", result.confidence)
        self.introspection.current_state = ExecutionState.IDLE

        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "method": result.method,
            "trace": result.trace,
            "from_cache": False,
            "elapsed": result.elapsed_seconds,
            "domain": result.domain.name,
        }

    # ── Batch processing (e.g. ARC-AGI evaluation) ──────────────────

    def process_batch(
        self,
        queries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Process a batch of tasks (e.g. ARC-AGI evaluation set)."""
        results = []
        for item in queries:
            query = item.get("query", item.get("question", ""))
            context = item.get("context", item.get("task", None))
            result = self.process(query, context=context)
            results.append(result)
        return results

    # ── Convenience: ARC-AGI evaluation ─────────────────────────────

    def evaluate_arc(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate on ARC-AGI format tasks.

        Each task should have ``train`` and ``test`` keys with
        ``input`` and ``output`` grid pairs.
        """
        if self.reasoning.arc_solver is None:
            return {"error": "ARC solver not available", "score": 0.0}

        from ..intelligence.arc_solver import ArcTask

        correct = 0
        total = 0

        for task_dict in tasks:
            task = ArcTask.from_dict(task_dict)
            result = self.reasoning.arc_solver.solve(task)
            if result.predictions is not None and task.test_outputs is not None:
                # Check predictions against ground truth
                for pred, expected in zip(result.predictions, task.test_outputs):
                    total += 1
                    if pred is not None and np.array_equal(np.asarray(pred), np.asarray(expected)):
                        correct += 1
            elif task.test_outputs is not None:
                total += len(task.test_outputs)

        return {
            "correct": correct,
            "total": total,
            "score": correct / total if total > 0 else 0.0,
        }

    # ── Diagnostic / debug helpers ──────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Get current kernel state for debugging."""
        return {
            "execution_state": self.introspection.current_state.name,
            "steps_taken": self.introspection.steps_taken,
            "strategies_tried": self.introspection.strategies_tried,
            "working_memory_keys": self.working_memory.keys(),
            "skill_library_size": self.skill_library.size,
            "is_stuck": self.introspection.is_stuck(),
        }

    def reset(self) -> None:
        """Reset kernel state (not skill library)."""
        self.working_memory.clear()
        self.introspection = IntrospectionState()
