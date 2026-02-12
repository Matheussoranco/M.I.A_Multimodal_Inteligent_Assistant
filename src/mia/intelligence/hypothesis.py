"""
Hypothesis Testing Framework
==============================

Provides structured hypothesis generation, testing against evidence,
and ranking by parsimony / support.  Used by the ARC solver, cognitive
kernel, and reasoning engine to formalise the guess-and-check loop.

A *Hypothesis* is any callable that maps an input to a predicted output.
The framework tests each hypothesis against known (input, output) pairs,
scores them, ranks by correctness and complexity, and selects the best.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

I = TypeVar("I")  # input type
O = TypeVar("O")  # output type


@dataclass
class Hypothesis(Generic[I, O]):
    """A proposed explanation mapping inputs → outputs."""

    name: str
    predict: Callable[[I], O]
    complexity: int = 1  # lower = simpler (Occam)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence(Generic[I, O]):
    """An (input, expected_output) pair used to test hypotheses."""

    input: I
    expected: O
    weight: float = 1.0  # importance of this evidence
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of testing a hypothesis against a set of evidence."""

    hypothesis_name: str
    correct: int = 0
    incorrect: int = 0
    errors: int = 0
    weighted_score: float = 0.0
    total_weight: float = 0.0
    complexity: int = 1
    time_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        total = self.correct + self.incorrect + self.errors
        return self.correct / max(total, 1)

    @property
    def parsimony_score(self) -> float:
        """Score balancing accuracy and simplicity (Occam's razor)."""
        return self.accuracy - 0.01 * self.complexity

    @property
    def perfect(self) -> bool:
        return self.incorrect == 0 and self.errors == 0 and self.correct > 0


class HypothesisTester(Generic[I, O]):
    """Test, rank, and select hypotheses against evidence.

    Usage::

        tester = HypothesisTester(match_fn=np.array_equal)
        tester.add_evidence(Evidence(input=grid_in, expected=grid_out))
        tester.add_hypothesis(Hypothesis("rotate_cw", rotate_cw))
        results = tester.test_all()
        best = tester.select_best()
    """

    def __init__(
        self,
        match_fn: Optional[Callable[[O, O], bool]] = None,
    ) -> None:
        self.hypotheses: List[Hypothesis[I, O]] = []
        self.evidence: List[Evidence[I, O]] = []
        self.results: List[TestResult] = []
        self._match = match_fn or (lambda a, b: a == b)

    def add_hypothesis(self, h: Hypothesis[I, O]) -> None:
        self.hypotheses.append(h)

    def add_evidence(self, e: Evidence[I, O]) -> None:
        self.evidence.append(e)

    def test_one(self, h: Hypothesis[I, O]) -> TestResult:
        """Test a single hypothesis against all evidence."""
        t0 = time.time()
        result = TestResult(
            hypothesis_name=h.name,
            complexity=h.complexity,
        )
        for ev in self.evidence:
            try:
                predicted = h.predict(ev.input)
                if self._match(predicted, ev.expected):
                    result.correct += 1
                    result.weighted_score += ev.weight
                else:
                    result.incorrect += 1
            except Exception:
                result.errors += 1
            result.total_weight += ev.weight
        result.time_seconds = time.time() - t0
        return result

    def test_all(self) -> List[TestResult]:
        """Test all hypotheses and store results."""
        self.results = [self.test_one(h) for h in self.hypotheses]
        return self.results

    def rank(self) -> List[TestResult]:
        """Rank results: perfect first, then by parsimony score."""
        if not self.results:
            self.test_all()
        return sorted(
            self.results,
            key=lambda r: (r.perfect, r.parsimony_score),
            reverse=True,
        )

    def select_best(self) -> Optional[Hypothesis[I, O]]:
        """Return the best hypothesis, or None if none are correct."""
        ranked = self.rank()
        if not ranked or ranked[0].accuracy == 0:
            return None
        best_name = ranked[0].hypothesis_name
        for h in self.hypotheses:
            if h.name == best_name:
                return h
        return None

    def get_perfect(self) -> List[Hypothesis[I, O]]:
        """Return all hypotheses that pass all evidence perfectly."""
        if not self.results:
            self.test_all()
        perfect_names = {r.hypothesis_name for r in self.results if r.perfect}
        return [h for h in self.hypotheses if h.name in perfect_names]

    def prune_failing(self, min_accuracy: float = 0.5) -> int:
        """Remove hypotheses below a minimum accuracy.  Returns count removed."""
        if not self.results:
            self.test_all()
        keep_names = {r.hypothesis_name for r in self.results if r.accuracy >= min_accuracy}
        before = len(self.hypotheses)
        self.hypotheses = [h for h in self.hypotheses if h.name in keep_names]
        removed = before - len(self.hypotheses)
        if removed:
            logger.debug("Pruned %d failing hypotheses", removed)
        return removed

    def refine(
        self,
        generator: Callable[[Hypothesis[I, O], List[Evidence[I, O]]], List[Hypothesis[I, O]]],
    ) -> List[Hypothesis[I, O]]:
        """Generate new hypotheses by refining existing ones that partially pass.

        *generator* receives a hypothesis and the evidence it failed on,
        and returns new candidate hypotheses.
        """
        if not self.results:
            self.test_all()

        new_hypotheses: List[Hypothesis[I, O]] = []
        for h, res in zip(self.hypotheses, self.results):
            if 0 < res.accuracy < 1.0:
                # Gather failed evidence
                failed_ev: List[Evidence[I, O]] = []
                for ev in self.evidence:
                    try:
                        pred = h.predict(ev.input)
                        if not self._match(pred, ev.expected):
                            failed_ev.append(ev)
                    except Exception:
                        failed_ev.append(ev)
                if failed_ev:
                    new_hypotheses.extend(generator(h, failed_ev))

        for nh in new_hypotheses:
            self.add_hypothesis(nh)

        return new_hypotheses
