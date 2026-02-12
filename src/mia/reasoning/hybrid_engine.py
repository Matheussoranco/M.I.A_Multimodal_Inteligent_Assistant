"""
Hybrid Reasoning Engine v2 for M.I.A
======================================

Combines *algorithmic* reasoning (constraint solving, SAT, logic,
graph algorithms) with *LLM-augmented* reasoning (CoT, ToT, ReAct)
and *code-level intelligence* modules (program synthesis, pattern
recognition, hypothesis testing).

Key design principle: **try deterministic/algorithmic first, fall
back to LLM only when the task genuinely requires natural-language
understanding or creative generation.**

Architecture
------------
1. Task classifier — detects reasoning type without LLM
2. Algorithmic layer — runs pure algorithms (no LLM)
3. Intelligence layer — grid DSL, program synthesis, patterns
4. LLM-assisted layer — uses LLM only when needed
5. Ensemble voting — combines multiple strategy results
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Task classification (heuristic, no LLM needed)
# ═══════════════════════════════════════════════════════════════════════════


class TaskDomain(Enum):
    MATH_ARITHMETIC = auto()
    MATH_ALGEBRA = auto()
    LOGIC_PROPOSITIONAL = auto()
    LOGIC_CONSTRAINT = auto()
    SEQUENCE_PATTERN = auto()
    GRID_TRANSFORM = auto()       # ARC-AGI type
    ANALOGY = auto()
    GRAPH_PATH = auto()
    CODE_GENERATION = auto()
    NATURAL_LANGUAGE = auto()      # fallback → requires LLM
    UNKNOWN = auto()


_DOMAIN_PATTERNS: Dict[TaskDomain, List[str]] = {
    TaskDomain.MATH_ARITHMETIC: [
        r"\d+\s*[\+\-\*\/\%\^]\s*\d+",
        r"calculate|compute|evaluate|what is \d|how much is",
    ],
    TaskDomain.MATH_ALGEBRA: [
        r"solve|equation|variable|unknown|x\s*=|find\s*x",
        r"linear\s*system|simultaneous",
    ],
    TaskDomain.LOGIC_PROPOSITIONAL: [
        r"\bif\b.*\bthen\b", r"\band\b.*\bor\b", r"implies|entails|deduce",
        r"true|false|premises|conclusion|syllogism",
    ],
    TaskDomain.LOGIC_CONSTRAINT: [
        r"constraint|sudoku|puzzle|assign|satisfy|permutation.*constraint",
    ],
    TaskDomain.SEQUENCE_PATTERN: [
        r"next|sequence|pattern.*\d+.*\d+.*\d+|what comes after",
        r"fibonacci|arithmetic\s*progression|geometric|series",
        r"\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+",
    ],
    TaskDomain.GRID_TRANSFORM: [
        r"grid|matrix\s*transform|arc[\s-]agi|rotate.*grid|flip.*grid",
        r"2d\s*array|pixel|tile|cell\s*color",
    ],
    TaskDomain.ANALOGY: [
        r"is\s*to.*as.*is\s*to", r"analogy|analogous|proportion",
        r"A\s*:\s*B\s*::\s*C",
    ],
    TaskDomain.GRAPH_PATH: [
        r"shortest\s*path|graph|route|distance\s*between|connected",
        r"node|edge|vertex|dijkstra",
    ],
    TaskDomain.CODE_GENERATION: [
        r"write\s*(a\s*)?function|implement|code|script|program",
        r"class\s+\w+|def\s+\w+|algorithm\s+for",
    ],
}


def classify_task(query: str) -> TaskDomain:
    """Rule-based task classification — no LLM call."""
    query_lower = query.lower()
    scores: Dict[TaskDomain, int] = {}

    for domain, patterns in _DOMAIN_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, query_lower))
        if score > 0:
            scores[domain] = score

    if not scores:
        return TaskDomain.NATURAL_LANGUAGE

    return max(scores, key=lambda d: scores[d])


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ReasoningResult:
    """Outcome from the hybrid reasoning engine."""
    answer: Any
    confidence: float          # 0.0–1.0
    method: str                # which strategy produced this
    domain: TaskDomain
    trace: List[str] = field(default_factory=list)   # step-by-step log
    alternatives: List[Any] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Reasoning Engine
# ═══════════════════════════════════════════════════════════════════════════


class HybridReasoningEngine:
    """Orchestrates algorithmic + intelligence + LLM reasoning.

    Usage::

        engine = HybridReasoningEngine(llm=my_llm)
        result = engine.reason("What is 17 * 23?")
        print(result.answer, result.confidence)
    """

    def __init__(
        self,
        llm: Any = None,
        *,
        prefer_algorithmic: bool = True,
        ensemble_threshold: int = 3,
        time_limit: float = 30.0,
    ) -> None:
        self.llm = llm
        self.prefer_algorithmic = prefer_algorithmic
        self.ensemble_threshold = ensemble_threshold
        self.time_limit = time_limit

        # Lazy-loaded modules
        self._algorithmic: Any = None
        self._arc_solver: Any = None
        self._synthesiser: Any = None
        self._patterns: Any = None
        self._hypothesis_tester: Any = None

    # ── Lazy loaders ────────────────────────────────────────────────

    @property
    def algorithmic(self):
        if self._algorithmic is None:
            from .algorithmic import AlgorithmicReasoner
            self._algorithmic = AlgorithmicReasoner()
        return self._algorithmic

    @property
    def arc_solver(self):
        if self._arc_solver is None:
            try:
                from ..intelligence.arc_solver import ArcSolver
                self._arc_solver = ArcSolver()
            except ImportError:
                self._arc_solver = False
        return self._arc_solver if self._arc_solver is not False else None

    @property
    def synthesiser(self):
        if self._synthesiser is None:
            try:
                from ..intelligence.program_synthesis import ProgramSynthesiser
                self._synthesiser = ProgramSynthesiser()
            except ImportError:
                self._synthesiser = False
        return self._synthesiser if self._synthesiser is not False else None

    @property
    def patterns(self):
        if self._patterns is None:
            try:
                from ..intelligence import patterns as pat_mod
                self._patterns = pat_mod
            except ImportError:
                self._patterns = False
        return self._patterns if self._patterns is not False else None

    @property
    def hypothesis_tester(self):
        if self._hypothesis_tester is None:
            try:
                from ..intelligence.hypothesis import HypothesisTester
                self._hypothesis_tester = HypothesisTester()
            except ImportError:
                self._hypothesis_tester = False
        return self._hypothesis_tester if self._hypothesis_tester is not False else None

    # ── Main entry point ────────────────────────────────────────────

    def reason(
        self,
        query: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        domain_hint: Optional[TaskDomain] = None,
    ) -> ReasoningResult:
        """Reason about *query*, automatically selecting the best method."""
        t0 = time.time()
        domain = domain_hint or classify_task(query)
        trace: List[str] = [f"Classified domain: {domain.name}"]

        # Dispatch to domain-specific handler
        handler = self._handlers.get(domain, self._handle_natural_language)
        result = handler(self, query, context, trace)

        result.domain = domain
        result.elapsed_seconds = time.time() - t0
        return result

    # ── Domain handlers ─────────────────────────────────────────────

    def _handle_arithmetic(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Pure eval for safe arithmetic expressions."""
        # Extract numeric expression — try multiple patterns
        candidates = [
            re.search(r"(?:what is|compute|calculate|evaluate)\s+([\d\.\s\+\-\*\/\%\(\)\^]+)", query, re.I),
            re.search(r"([\d\.]+\s*[\+\-\*\/\%\^]\s*[\d\.\s\+\-\*\/\%\(\)\^]+)", query),
        ]
        for expr_match in candidates:
            if expr_match:
                expr = expr_match.group(1).strip().replace("^", "**")
                try:
                    result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
                    trace.append(f"Evaluated: {expr} = {result}")
                    return ReasoningResult(
                        answer=result, confidence=1.0, method="arithmetic_eval",
                        domain=TaskDomain.MATH_ARITHMETIC, trace=trace,
                    )
                except Exception as e:
                    trace.append(f"Eval failed for '{expr}': {e}")

        return self._fallback_llm(query, trace, "arithmetic_llm")

    def _handle_algebra(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Try symbolic linear solve, else fall back to LLM."""
        # Simple 1-variable linear equation: ax + b = c
        match = re.search(
            r"(-?\d*\.?\d*)\s*\*?\s*x\s*([+\-]\s*\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)",
            query.lower(),
        )
        if match:
            a_str, b_str, c_str = match.groups()
            a = float(a_str) if a_str and a_str not in ("", "+", "-") else (
                -1.0 if a_str == "-" else 1.0
            )
            b = float(b_str.replace(" ", ""))
            c = float(c_str)
            x = (c - b) / a
            trace.append(f"Solved {a}x + {b} = {c}  →  x = {x}")
            return ReasoningResult(
                answer=x, confidence=1.0, method="linear_solver",
                domain=TaskDomain.MATH_ALGEBRA, trace=trace,
            )

        return self._fallback_llm(query, trace, "algebra_llm")

    def _handle_logic(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Use the LogicEngine for propositional reasoning."""
        engine = self.algorithmic.logic

        # Try to parse "if A then B" style rules from the query
        rule_pattern = re.findall(
            r"if\s+(.+?)\s+then\s+(.+?)(?:\.|;|$)", query, re.IGNORECASE,
        )
        for premises_str, conclusion in rule_pattern:
            premises = [p.strip().lower() for p in re.split(r"\s+and\s+", premises_str)]
            engine.add_rule(premises, conclusion.strip().lower())
            trace.append(f"Rule: {premises} → {conclusion.strip().lower()}")

        # Try to parse known facts
        fact_pattern = re.findall(
            r"(?:we know|it is true|given)(?:\s+that)?\s+(.+?)(?:\.|;|$)",
            query, re.IGNORECASE,
        )
        for fact_str in fact_pattern:
            for fact in re.split(r"\s+and\s+", fact_str):
                engine.tell(fact.strip().lower())
                trace.append(f"Fact: {fact.strip().lower()}")

        # Try to parse the query goal
        goal_match = re.search(
            r"(?:is|can we conclude|does it follow|prove)\s+(?:that\s+)?(.+?)(?:\?|$)",
            query, re.IGNORECASE,
        )
        if goal_match:
            goal = goal_match.group(1).strip().lower()
            result = engine.backward_chain(goal)
            proof = engine.explain(goal)
            trace.extend(proof)
            trace.append(f"Conclusion: {goal} is {'TRUE' if result else 'UNDETERMINED'}")
            return ReasoningResult(
                answer=result, confidence=0.95 if result else 0.7,
                method="logic_engine", domain=TaskDomain.LOGIC_PROPOSITIONAL,
                trace=trace,
            )

        # Forward-chain and report all derived facts
        all_facts = engine.forward_chain()
        trace.append(f"Derived facts: {all_facts}")
        return ReasoningResult(
            answer=all_facts, confidence=0.8, method="forward_chain",
            domain=TaskDomain.LOGIC_PROPOSITIONAL, trace=trace,
        )

    def _handle_constraint(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """CSP solving — if context provides structure, solve it algorithmically."""
        if ctx and "csp" in ctx:
            csp_data = ctx["csp"]
            solutions = self.algorithmic.solve_csp(
                csp_data["variables"],
                csp_data["domains"],
                csp_data["constraints"],
            )
            trace.append(f"CSP solutions found: {len(solutions)}")
            return ReasoningResult(
                answer=solutions[0] if solutions else None,
                confidence=1.0 if solutions else 0.0,
                method="csp_solver", domain=TaskDomain.LOGIC_CONSTRAINT,
                trace=trace, alternatives=solutions[1:],
            )
        return self._fallback_llm(query, trace, "constraint_llm")

    def _handle_sequence(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Detect and predict numerical sequences algorithmically."""
        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
        if len(numbers) >= 3:
            predictions = self.algorithmic.sequence_predict(numbers)
            if predictions:
                trace.append(f"Sequence: {numbers} → predicted: {predictions}")
                return ReasoningResult(
                    answer=predictions[0], confidence=0.9,
                    method="sequence_solver", domain=TaskDomain.SEQUENCE_PATTERN,
                    trace=trace, alternatives=predictions[1:],
                )
        return self._fallback_llm(query, trace, "sequence_llm")

    def _handle_grid(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Route to ARC solver or program synthesis."""
        if ctx and "task" in ctx and self.arc_solver:
            from ..intelligence.arc_solver import ArcTask

            task = ArcTask.from_dict(ctx["task"])
            answer, strategy = self.arc_solver.solve(task)
            verified = answer is not None
            trace.append(f"ARC strategy: {strategy}, verified: {verified}")
            return ReasoningResult(
                answer=answer, confidence=0.95 if verified else 0.3,
                method=f"arc_{strategy}", domain=TaskDomain.GRID_TRANSFORM,
                trace=trace,
            )

        if ctx and "examples" in ctx and self.synthesiser:
            programmes = self.synthesiser.synthesise(ctx["examples"])
            if programmes:
                best = programmes[0]
                trace.append(f"Synthesised program: {best}")
                return ReasoningResult(
                    answer=best, confidence=0.85,
                    method="program_synthesis", domain=TaskDomain.GRID_TRANSFORM,
                    trace=trace,
                )

        return self._fallback_llm(query, trace, "grid_llm")

    def _handle_analogy(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Numeric analogy solving."""
        numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", query)]
        if len(numbers) >= 3:
            candidates = self.algorithmic.numeric_analogy(*numbers[:3])
            trace.append(f"Analogy {numbers[:3]} → candidates: {candidates}")
            return ReasoningResult(
                answer=candidates[0] if candidates else None,
                confidence=0.8 if candidates else 0.0,
                method="numeric_analogy", domain=TaskDomain.ANALOGY,
                trace=trace, alternatives=candidates[1:],
            )
        return self._fallback_llm(query, trace, "analogy_llm")

    def _handle_graph(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Graph path reasoning via Dijkstra."""
        if ctx and "edges" in ctx and "start" in ctx and "end" in ctx:
            path = self.algorithmic.shortest_path(
                ctx["edges"], ctx["start"], ctx["end"],
            )
            trace.append(f"Shortest path: {path}")
            return ReasoningResult(
                answer=path, confidence=1.0 if path else 0.0,
                method="dijkstra", domain=TaskDomain.GRAPH_PATH,
                trace=trace,
            )
        return self._fallback_llm(query, trace, "graph_llm")

    def _handle_natural_language(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Genuine NL tasks → delegate to LLM."""
        return self._fallback_llm(query, trace, "llm_direct")

    def _handle_code(
        self, query: str, ctx: Optional[Dict], trace: List[str],
    ) -> ReasoningResult:
        """Code generation → LLM is genuinely needed."""
        return self._fallback_llm(query, trace, "code_llm")

    # ── LLM fallback ─────────────────────────────────────────────

    def _fallback_llm(
        self, query: str, trace: List[str], method: str,
    ) -> ReasoningResult:
        """Use LLM as a last resort."""
        if self.llm is None:
            trace.append("No LLM available — returning empty.")
            return ReasoningResult(
                answer=None, confidence=0.0, method=method,
                domain=TaskDomain.UNKNOWN, trace=trace,
            )
        trace.append("Falling back to LLM")
        try:
            if hasattr(self.llm, "query"):
                answer = self.llm.query(query)
            elif hasattr(self.llm, "query_model"):
                answer = self.llm.query_model(query)
            else:
                answer = None
            trace.append(f"LLM answer length: {len(str(answer or ''))}")
            return ReasoningResult(
                answer=answer, confidence=0.6, method=method,
                domain=TaskDomain.NATURAL_LANGUAGE, trace=trace,
            )
        except Exception as e:
            trace.append(f"LLM error: {e}")
            return ReasoningResult(
                answer=None, confidence=0.0, method=method,
                domain=TaskDomain.UNKNOWN, trace=trace,
            )

    # ── Ensemble reasoning ─────────────────────────────────────────

    def reason_ensemble(
        self,
        query: str,
        strategies: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Run multiple strategies and vote on the best answer.

        This is the SOTA approach: combine algorithmic precision
        with LLM creativity, then pick the answer with highest
        cross-strategy agreement.
        """
        results: List[ReasoningResult] = []

        # Always try algorithmic first
        results.append(self.reason(query, context=context))

        # Try sequence / analogy if numbers present
        numbers = re.findall(r"-?\d+\.?\d*", query)
        if len(numbers) >= 3:
            results.append(self.reason(
                query, context=context, domain_hint=TaskDomain.SEQUENCE_PATTERN,
            ))
            results.append(self.reason(
                query, context=context, domain_hint=TaskDomain.ANALOGY,
            ))

        # Always include LLM if available
        if self.llm:
            results.append(self.reason(
                query, context=context, domain_hint=TaskDomain.NATURAL_LANGUAGE,
            ))

        # Pick highest confidence
        best = max(results, key=lambda r: r.confidence)
        best.trace.insert(0, f"Ensemble: {len(results)} strategies evaluated")
        best.alternatives = [
            r.answer for r in results if r.answer != best.answer
        ]
        return best

    # ── Handler dispatch table ───────────────────────────────────

    _handlers: Dict[TaskDomain, Callable] = {
        TaskDomain.MATH_ARITHMETIC: _handle_arithmetic,
        TaskDomain.MATH_ALGEBRA: _handle_algebra,
        TaskDomain.LOGIC_PROPOSITIONAL: _handle_logic,
        TaskDomain.LOGIC_CONSTRAINT: _handle_constraint,
        TaskDomain.SEQUENCE_PATTERN: _handle_sequence,
        TaskDomain.GRID_TRANSFORM: _handle_grid,
        TaskDomain.ANALOGY: _handle_analogy,
        TaskDomain.GRAPH_PATH: _handle_graph,
        TaskDomain.CODE_GENERATION: _handle_code,
        TaskDomain.NATURAL_LANGUAGE: _handle_natural_language,
        TaskDomain.UNKNOWN: _handle_natural_language,
    }
