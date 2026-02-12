"""
Program Synthesis Engine — Example-Guided Search over DSL Programs
===================================================================

Given (input, output) example pairs and a library of composable DSL
primitives, **synthesise** the shortest program that transforms every
input into its corresponding output.

Strategies (tried in order of increasing cost):
1. **Single-primitive scan** — try every 0-arity primitive alone.
2. **Parameterised single-primitive** — enumerate parameter combos.
3. **Two-step composition** — all pairs of primitives.
4. **Beam search over programs** — iteratively extend promising partial
   programs by appending primitives.
5. **LLM-guided synthesis** — use an LLM to propose candidate programs
   based on grid descriptions (optional, hybrid approach).

The engine is designed for ARC-AGI but works on any domain whose
primitives are registered in ``grid_dsl.PRIMITIVES``.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .grid_dsl import (
    Grid,
    PRIMITIVES,
    Program,
    PrimitiveSpec,
    ZERO_ARITY_NAMES,
    compose,
    diff_count,
    grids_equal,
    program_distance,
    program_score,
    verify_program,
    background_color,
    palette,
    find_objects,
    shape_relation,
    color_mapping,
)
from .hypothesis import Evidence, Hypothesis, HypothesisTester

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SynthesisResult:
    """Outcome of a synthesis attempt."""

    program: Optional[Program] = None
    score: float = 0.0
    time_seconds: float = 0.0
    programs_tested: int = 0
    strategy: str = ""
    verified: bool = False

    @property
    def success(self) -> bool:
        return self.verified and self.program is not None


@dataclass
class SynthesisConfig:
    """Tuning knobs for the synthesiser."""

    max_program_length: int = 4
    beam_width: int = 30
    time_limit: float = 60.0
    max_candidates: int = 50_000
    param_sample_limit: int = 8
    enable_llm_hints: bool = False

    # Colour values to try for parameterised primitives
    color_range: Tuple[int, ...] = tuple(range(10))
    int_range: Tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _extract_task_colors(
    examples: List[Tuple[Grid, Grid]],
) -> Set[int]:
    """Gather all colours that appear in any example."""
    colors: Set[int] = set()
    for inp, out in examples:
        colors |= set(int(v) for v in np.unique(inp))
        colors |= set(int(v) for v in np.unique(out))
    return colors


def _param_combos(
    spec: PrimitiveSpec, task_colors: Set[int], config: SynthesisConfig
) -> List[Dict[str, Any]]:
    """Enumerate parameter combinations for a primitive, bounded."""
    schema = spec.param_schema
    if not schema:
        return [{}]

    axes: List[List[Any]] = []
    names: List[str] = []

    for pname, pspec in schema.items():
        names.append(pname)
        ptype = pspec.get("type", "int")
        if ptype == "color":
            axes.append(sorted(task_colors))
        elif ptype == "enum":
            axes.append(pspec.get("values", []))
        elif ptype == "int":
            rng = pspec.get("range")
            if rng:
                axes.append(list(range(rng[0], rng[1] + 1)))
            else:
                axes.append(list(config.int_range))
        else:
            axes.append([0, 1, 2])

    combos: List[Dict[str, Any]] = []
    for vals in itertools.islice(itertools.product(*axes), config.max_candidates):
        combos.append(dict(zip(names, vals)))
        if len(combos) >= config.param_sample_limit * len(names):
            break
    return combos


# ═══════════════════════════════════════════════════════════════════════
# Core Synthesis Engine
# ═══════════════════════════════════════════════════════════════════════


class ProgramSynthesiser:
    """Synthesise DSL programs from (input, output) examples.

    Usage::

        synth = ProgramSynthesiser()
        result = synth.synthesise([(grid_in, grid_out), ...])
        if result.success:
            output = compose(test_input, result.program)
    """

    def __init__(self, config: Optional[SynthesisConfig] = None) -> None:
        self.config = config or SynthesisConfig()
        self.primitives = PRIMITIVES
        self._stats = {"tested": 0, "strategy": ""}

    def synthesise(
        self,
        examples: List[Tuple[Grid, Grid]],
        test_inputs: Optional[List[Grid]] = None,
    ) -> SynthesisResult:
        """Try all strategies in order. Return first verified program."""
        t0 = time.time()
        self._stats = {"tested": 0, "strategy": ""}
        task_colors = _extract_task_colors(examples)

        strategies: List[Tuple[str, Callable]] = [
            ("identity_check", lambda: self._try_identity(examples)),
            ("color_map", lambda: self._try_color_mapping(examples, task_colors)),
            ("single_zero_arity", lambda: self._try_single_zero(examples)),
            ("single_parameterised", lambda: self._try_single_param(examples, task_colors)),
            ("two_step", lambda: self._try_two_step(examples, task_colors)),
            ("beam_search", lambda: self._try_beam(examples, task_colors)),
        ]

        for name, strategy_fn in strategies:
            if time.time() - t0 > self.config.time_limit:
                break
            self._stats["strategy"] = name
            result = strategy_fn()
            if result and result.success:
                result.time_seconds = time.time() - t0
                result.programs_tested = self._stats["tested"]
                result.strategy = name
                logger.info(
                    "Synthesis succeeded: strategy=%s program=%s tested=%d time=%.2fs",
                    name, result.program, self._stats["tested"], result.time_seconds,
                )
                return result

        return SynthesisResult(
            time_seconds=time.time() - t0,
            programs_tested=self._stats["tested"],
            strategy="exhausted",
        )

    # ── Strategy 0: Identity ────────────────────────────────────────

    def _try_identity(
        self, examples: List[Tuple[Grid, Grid]]
    ) -> Optional[SynthesisResult]:
        """Check if output == input for every example."""
        if all(grids_equal(i, o) for i, o in examples):
            return SynthesisResult(
                program=[], score=1.0, strategy="identity", verified=True
            )
        return None

    # ── Strategy 1: Pure colour mapping ─────────────────────────────

    def _try_color_mapping(
        self,
        examples: List[Tuple[Grid, Grid]],
        task_colors: Set[int],
    ) -> Optional[SynthesisResult]:
        """Check if the transformation is a pure per-pixel colour remap."""
        from .grid_dsl import color_mapping as _cmap, replace_color

        mappings = [_cmap(i, o) for i, o in examples]
        if any(m is None for m in mappings):
            return None

        # All mappings should agree
        ref = mappings[0]
        if not all(m == ref for m in mappings):
            return None

        # Build program from replace_color steps
        program: Program = []
        for old, new in ref.items():  # type: ignore[union-attr]
            if old != new:
                program.append(("replace_color", {"old": old, "new": new}))

        self._stats["tested"] += 1
        if verify_program(program, examples):
            return SynthesisResult(
                program=program, score=1.0, strategy="color_map", verified=True
            )
        return None

    # ── Strategy 2: Single zero-arity primitive ─────────────────────

    def _try_single_zero(
        self, examples: List[Tuple[Grid, Grid]]
    ) -> Optional[SynthesisResult]:
        for name in ZERO_ARITY_NAMES:
            self._stats["tested"] += 1
            prog: Program = [(name, {})]
            if verify_program(prog, examples):
                return SynthesisResult(
                    program=prog, score=1.0, strategy="single_zero", verified=True
                )
        return None

    # ── Strategy 3: Single primitive with parameters ────────────────

    def _try_single_param(
        self,
        examples: List[Tuple[Grid, Grid]],
        task_colors: Set[int],
    ) -> Optional[SynthesisResult]:
        for name, spec in self.primitives.items():
            if spec.arity == 0:
                continue
            for params in _param_combos(spec, task_colors, self.config):
                self._stats["tested"] += 1
                prog: Program = [(name, params)]
                if verify_program(prog, examples):
                    return SynthesisResult(
                        program=prog, score=1.0, strategy="single_param", verified=True
                    )
                if self._stats["tested"] > self.config.max_candidates:
                    return None
        return None

    # ── Strategy 4: Two-step composition ────────────────────────────

    def _try_two_step(
        self,
        examples: List[Tuple[Grid, Grid]],
        task_colors: Set[int],
    ) -> Optional[SynthesisResult]:
        # First try pairs of zero-arity
        for n1 in ZERO_ARITY_NAMES:
            for n2 in ZERO_ARITY_NAMES:
                self._stats["tested"] += 1
                prog: Program = [(n1, {}), (n2, {})]
                if verify_program(prog, examples):
                    return SynthesisResult(
                        program=prog, score=1.0, strategy="two_step_zero", verified=True
                    )
            if self._stats["tested"] > self.config.max_candidates // 2:
                break

        # Then try zero-arity + parameterised
        for n1 in ZERO_ARITY_NAMES:
            for n2, spec2 in self.primitives.items():
                if spec2.arity == 0:
                    continue
                for params in _param_combos(spec2, task_colors, self.config)[:4]:
                    self._stats["tested"] += 1
                    prog = [(n1, {}), (n2, params)]
                    if verify_program(prog, examples):
                        return SynthesisResult(
                            program=prog, score=1.0, strategy="two_step_mixed", verified=True
                        )
                if self._stats["tested"] > self.config.max_candidates:
                    return None

        return None

    # ── Strategy 5: Beam search ─────────────────────────────────────

    def _try_beam(
        self,
        examples: List[Tuple[Grid, Grid]],
        task_colors: Set[int],
    ) -> Optional[SynthesisResult]:
        """Beam search over program space, keeping best partial programs."""
        beam: List[Tuple[float, Program]] = [(0.0, [])]
        best_result: Optional[SynthesisResult] = None
        best_score = -1.0

        for length in range(1, self.config.max_program_length + 1):
            next_beam: List[Tuple[float, Program]] = []

            for _, base_prog in beam:
                # Extend with each primitive
                for name, spec in self.primitives.items():
                    if spec.arity == 0:
                        param_sets = [{}]
                    else:
                        param_sets = _param_combos(spec, task_colors, self.config)[:3]

                    for params in param_sets:
                        self._stats["tested"] += 1
                        candidate = base_prog + [(name, params)]

                        if verify_program(candidate, examples):
                            return SynthesisResult(
                                program=candidate,
                                score=1.0,
                                strategy="beam",
                                verified=True,
                            )

                        sc = program_score(candidate, examples)
                        dist = program_distance(candidate, examples)
                        combined = sc - 0.001 * dist - 0.01 * len(candidate)
                        next_beam.append((combined, candidate))

                        if sc > best_score:
                            best_score = sc
                            best_result = SynthesisResult(
                                program=candidate, score=sc, strategy="beam"
                            )

                        if self._stats["tested"] > self.config.max_candidates:
                            return best_result

            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[: self.config.beam_width]

        return best_result

    # ── Hypothesis-tester integration ───────────────────────────────

    def as_hypotheses(
        self,
        examples: List[Tuple[Grid, Grid]],
        max_programs: int = 200,
    ) -> HypothesisTester[Grid, Grid]:
        """Generate hypotheses from DSL programs and wrap in a tester."""
        tester: HypothesisTester[Grid, Grid] = HypothesisTester(
            match_fn=grids_equal
        )
        for inp, out in examples:
            tester.add_evidence(Evidence(input=inp, expected=out))

        task_colors = _extract_task_colors(examples)
        count = 0

        # Single primitives
        for name in ZERO_ARITY_NAMES:
            prog: Program = [(name, {})]
            tester.add_hypothesis(Hypothesis(
                name=name,
                predict=lambda g, p=prog: compose(g.copy(), p),
                complexity=1,
                description=f"Apply {name}",
            ))
            count += 1

        # Parameterised
        for name, spec in self.primitives.items():
            if spec.arity == 0:
                continue
            for params in _param_combos(spec, task_colors, self.config)[:6]:
                prog = [(name, params)]
                label = f"{name}({params})"
                tester.add_hypothesis(Hypothesis(
                    name=label,
                    predict=lambda g, p=prog: compose(g.copy(), p),
                    complexity=1 + len(params),
                    description=label,
                ))
                count += 1
                if count >= max_programs:
                    return tester

        # Two-step zero-arity combos
        for n1 in ZERO_ARITY_NAMES:
            for n2 in ZERO_ARITY_NAMES:
                prog = [(n1, {}), (n2, {})]
                label = f"{n1} -> {n2}"
                tester.add_hypothesis(Hypothesis(
                    name=label,
                    predict=lambda g, p=prog: compose(g.copy(), p),
                    complexity=2,
                    description=label,
                ))
                count += 1
                if count >= max_programs:
                    return tester

        return tester


# ═══════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════


def synthesise_program(
    examples: List[Tuple[Grid, Grid]],
    time_limit: float = 60.0,
) -> SynthesisResult:
    """One-shot convenience wrapper."""
    config = SynthesisConfig(time_limit=time_limit)
    return ProgramSynthesiser(config).synthesise(examples)


def apply_synthesised(
    examples: List[Tuple[Grid, Grid]],
    test_input: Grid,
    time_limit: float = 60.0,
) -> Optional[Grid]:
    """Synthesise a program and apply it to a test input."""
    result = synthesise_program(examples, time_limit)
    if result.success and result.program is not None:
        return compose(test_input.copy(), result.program)
    return None
