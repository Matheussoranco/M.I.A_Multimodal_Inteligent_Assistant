"""
ARC-AGI Solver — Multi-Strategy Solver for Abstract Reasoning Corpus
=====================================================================

Combines all intelligence sub-modules into a unified solver that
attacks ARC-AGI tasks with multiple strategies in a tournament:

1. **Program synthesis** — search DSL program space
2. **Pattern matching** — detect task patterns and apply templates
3. **Analogical reasoning** — A:B :: C:? via transform detection
4. **Object-centric transforms** — operate on detected objects
5. **LLM-guided reasoning** — use LLM for hypothesis generation (optional)

The solver runs all applicable strategies, verifies each against the
training examples, and returns the first solution that passes all
examples (or the highest-scoring partial solution).

This module is the primary entry point for ARC-AGI benchmark evaluation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .grid_dsl import (
    Grid,
    Program,
    compose,
    crop,
    find_objects,
    flip_h,
    flip_v,
    grids_equal,
    overlay,
    rotate_180,
    rotate_cw,
    rotate_ccw,
    transpose,
    background_color,
    mirror_complete_h,
    mirror_complete_v,
    replace_color,
    gravity,
    tile,
    upscale,
    verify_program,
    program_score,
)
from .patterns import (
    GridFeatures,
    TaskPattern,
    analyse_task,
    extract_features,
    detect_periodicity,
    grid_analogy,
    induce_grid_rules,
    abstract_grid,
    compute_spatial_relations,
)
from .program_synthesis import (
    ProgramSynthesiser,
    SynthesisConfig,
    SynthesisResult,
)
from .hypothesis import Evidence, Hypothesis, HypothesisTester
from .search import beam_search, SearchResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ArcTask:
    """An ARC-AGI task with training and test examples."""

    task_id: str
    train: List[Tuple[Grid, Grid]]
    test_inputs: List[Grid]
    test_outputs: Optional[List[Grid]] = None  # ground truth (for eval)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], task_id: str = "unknown") -> "ArcTask":
        """Parse from the standard ARC JSON format."""
        train = [
            (np.array(ex["input"], dtype=int), np.array(ex["output"], dtype=int))
            for ex in data.get("train", [])
        ]
        test_inputs = [
            np.array(ex["input"], dtype=int) for ex in data.get("test", [])
        ]
        test_outputs = None
        if data.get("test") and "output" in data["test"][0]:
            test_outputs = [
                np.array(ex["output"], dtype=int) for ex in data["test"]
            ]
        return cls(
            task_id=task_id,
            train=train,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
        )


@dataclass
class SolverResult:
    """Result from the ARC solver."""

    task_id: str
    predictions: Sequence[Optional[Grid]]
    strategy: str = ""
    confidence: float = 0.0
    time_seconds: float = 0.0
    verified_on_train: bool = False
    program: Optional[Program] = None

    @property
    def success(self) -> bool:
        return self.verified_on_train and all(p is not None for p in self.predictions)


# ═══════════════════════════════════════════════════════════════════════
# Individual Strategies
# ═══════════════════════════════════════════════════════════════════════


def _strategy_synthesis(
    task: ArcTask, time_limit: float
) -> Optional[SolverResult]:
    """Strategy 1: Program synthesis over DSL."""
    config = SynthesisConfig(time_limit=time_limit)
    synth = ProgramSynthesiser(config)
    result = synth.synthesise(task.train)
    if result.success and result.program is not None:
        predictions = []
        for test_in in task.test_inputs:
            try:
                predictions.append(compose(test_in.copy(), result.program))
            except Exception:
                predictions.append(None)
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            strategy=f"synthesis_{result.strategy}",
            confidence=result.score,
            time_seconds=result.time_seconds,
            verified_on_train=True,
            program=result.program,
        )
    return None


def _strategy_analogy(
    task: ArcTask, time_limit: float
) -> Optional[SolverResult]:
    """Strategy 2: Analogical reasoning for each test input."""
    if len(task.train) < 1:
        return None
    t0 = time.time()
    a_in, a_out = task.train[0]

    # Verify on remaining training examples
    for inp, out in task.train[1:]:
        predicted = grid_analogy(a_in, a_out, inp)
        if predicted is None or not grids_equal(predicted, out):
            return None

    predictions = []
    for test_in in task.test_inputs:
        predicted = grid_analogy(a_in, a_out, test_in)
        predictions.append(predicted)

    if all(p is not None for p in predictions):
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            strategy="analogy",
            confidence=0.8,
            time_seconds=time.time() - t0,
            verified_on_train=True,
        )
    return None


def _strategy_object_transform(
    task: ArcTask, time_limit: float
) -> Optional[SolverResult]:
    """Strategy 3: Object-centric transforms.

    Detects objects in input, applies per-object transforms, and
    reconstructs the output.
    """
    t0 = time.time()
    pattern = analyse_task(task.train)

    # Only handle same-size tasks with same object count
    if pattern.size_relation != "same_size" or pattern.objects_change != "same_count":
        return None

    # For each training pair, figure out per-object colour changes
    # (simplest object-level transform)
    for inp, out in task.train:
        bg = background_color(inp)
        in_objs = find_objects(inp, bg)
        out_objs = find_objects(out, bg)
        if len(in_objs) != len(out_objs):
            return None

    # Try: keep structure, change per-object colour based on size ranking
    # This handles tasks where object colour depends on relative size
    def _apply_object_recolor(grid: Grid) -> Grid:
        bg = background_color(grid)
        objs = sorted(find_objects(grid, bg), key=lambda o: o.size, reverse=True)
        # Learn the colour assignment from training
        # Match by size rank
        ref_in, ref_out = task.train[0]
        ref_bg = background_color(ref_in)
        ref_in_objs = sorted(find_objects(ref_in, ref_bg), key=lambda o: o.size, reverse=True)
        ref_out_objs = sorted(find_objects(ref_out, ref_bg), key=lambda o: o.size, reverse=True)

        if len(ref_in_objs) != len(objs) or len(ref_in_objs) != len(ref_out_objs):
            return grid

        result = grid.copy()
        for obj, ref_out_obj in zip(objs, ref_out_objs):
            for r, c in obj.pixels:
                result[r, c] = ref_out_obj.color
        return result

    # Verify on training
    for inp, out in task.train:
        predicted = _apply_object_recolor(inp)
        if not grids_equal(predicted, out):
            return None

    predictions = [_apply_object_recolor(t) for t in task.test_inputs]
    return SolverResult(
        task_id=task.task_id,
        predictions=predictions,
        strategy="object_recolor",
        confidence=0.7,
        time_seconds=time.time() - t0,
        verified_on_train=True,
    )


def _strategy_tiling(
    task: ArcTask, time_limit: float
) -> Optional[SolverResult]:
    """Strategy 4: If output is a tiling/scaling of input."""
    t0 = time.time()
    pattern = analyse_task(task.train)

    if not pattern.size_relation.startswith("scale_"):
        return None

    # Extract scale factor
    try:
        factor = int(pattern.size_relation.split("_")[1].rstrip("x"))
    except (IndexError, ValueError):
        return None

    # Try tile
    prog_tile: Program = [("tile", {"nh": factor, "nw": factor})]
    if verify_program(prog_tile, task.train):
        predictions = [compose(t.copy(), prog_tile) for t in task.test_inputs]
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            strategy="tiling",
            confidence=0.9,
            time_seconds=time.time() - t0,
            verified_on_train=True,
            program=prog_tile,
        )

    # Try upscale
    prog_up: Program = [("upscale", {"factor": factor})]
    if verify_program(prog_up, task.train):
        predictions = [compose(t.copy(), prog_up) for t in task.test_inputs]
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            strategy="upscale",
            confidence=0.9,
            time_seconds=time.time() - t0,
            verified_on_train=True,
            program=prog_up,
        )

    return None


def _strategy_symmetry_completion(
    task: ArcTask, time_limit: float
) -> Optional[SolverResult]:
    """Strategy 5: Complete a partial symmetry."""
    t0 = time.time()
    pattern = analyse_task(task.train)

    if not pattern.symmetry_added:
        return None

    # Try horizontal symmetry completion
    prog_h: Program = [("mirror_complete_h", {})]
    if verify_program(prog_h, task.train):
        predictions = [compose(t.copy(), prog_h) for t in task.test_inputs]
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            strategy="symmetry_h",
            confidence=0.85,
            time_seconds=time.time() - t0,
            verified_on_train=True,
            program=prog_h,
        )

    # Try vertical
    prog_v: Program = [("mirror_complete_v", {})]
    if verify_program(prog_v, task.train):
        predictions = [compose(t.copy(), prog_v) for t in task.test_inputs]
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            strategy="symmetry_v",
            confidence=0.85,
            time_seconds=time.time() - t0,
            verified_on_train=True,
            program=prog_v,
        )

    return None


def _strategy_gravity(
    task: ArcTask, time_limit: float
) -> Optional[SolverResult]:
    """Strategy 6: Gravity in one direction."""
    t0 = time.time()

    for direction in ("down", "up", "left", "right"):
        prog: Program = [("gravity", {"direction": direction})]
        if verify_program(prog, task.train):
            predictions = [compose(t.copy(), prog) for t in task.test_inputs]
            return SolverResult(
                task_id=task.task_id,
                predictions=predictions,
                strategy=f"gravity_{direction}",
                confidence=0.9,
                time_seconds=time.time() - t0,
                verified_on_train=True,
                program=prog,
            )
    return None


def _strategy_crop_largest(
    task: ArcTask, time_limit: float
) -> Optional[SolverResult]:
    """Strategy 7: Output is the cropped largest object."""
    t0 = time.time()

    def _crop_largest(grid: Grid) -> Grid:
        bg = background_color(grid)
        objs = find_objects(grid, bg)
        if not objs:
            return crop(grid, bg)
        largest = max(objs, key=lambda o: o.size)
        return largest.as_grid(bg)

    # Verify
    for inp, out in task.train:
        predicted = _crop_largest(inp)
        if not grids_equal(predicted, out):
            return None

    predictions = [_crop_largest(t) for t in task.test_inputs]
    return SolverResult(
        task_id=task.task_id,
        predictions=predictions,
        strategy="crop_largest",
        confidence=0.8,
        time_seconds=time.time() - t0,
        verified_on_train=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Main Solver
# ═══════════════════════════════════════════════════════════════════════


class ArcSolver:
    """Multi-strategy ARC-AGI solver.

    Usage::

        solver = ArcSolver()
        result = solver.solve(task)
        if result.success:
            for pred in result.predictions:
                print(pred)
    """

    # Strategies ordered by speed (fast/simple first)
    DEFAULT_STRATEGIES = [
        ("analogy", _strategy_analogy),
        ("gravity", _strategy_gravity),
        ("symmetry", _strategy_symmetry_completion),
        ("tiling", _strategy_tiling),
        ("crop_largest", _strategy_crop_largest),
        ("object_transform", _strategy_object_transform),
        ("synthesis", _strategy_synthesis),
    ]

    def __init__(
        self,
        strategies: Optional[List[Tuple[str, Callable]]] = None,
        time_limit: float = 120.0,
        llm: Optional[Any] = None,
    ) -> None:
        self.strategies = strategies or self.DEFAULT_STRATEGIES
        self.time_limit = time_limit
        self.llm = llm  # optional LLM for hybrid approach
        self._last_analysis: Optional[TaskPattern] = None

    def solve(self, task: ArcTask) -> SolverResult:
        """Run all strategies and return the best result."""
        t0 = time.time()
        self._last_analysis = analyse_task(task.train)
        per_strategy_limit = self.time_limit / max(len(self.strategies), 1)

        best_result: Optional[SolverResult] = None

        for name, strategy_fn in self.strategies:
            elapsed = time.time() - t0
            if elapsed > self.time_limit:
                break
            remaining = self.time_limit - elapsed
            budget = min(per_strategy_limit, remaining)

            try:
                result = strategy_fn(task, budget)
                if result and result.verified_on_train:
                    result.time_seconds = time.time() - t0
                    logger.info(
                        "ARC solver: task=%s strategy=%s solved in %.2fs",
                        task.task_id, result.strategy, result.time_seconds,
                    )
                    return result
                if result and (best_result is None or result.confidence > best_result.confidence):
                    best_result = result
            except Exception as exc:
                logger.debug("Strategy %s failed: %s", name, exc)

        # Return best partial result or empty
        if best_result:
            best_result.time_seconds = time.time() - t0
            return best_result

        return SolverResult(
            task_id=task.task_id,
            predictions=[None] * len(task.test_inputs),
            strategy="none",
            time_seconds=time.time() - t0,
        )

    def solve_batch(
        self,
        tasks: List[ArcTask],
    ) -> Dict[str, SolverResult]:
        """Solve multiple tasks and return results keyed by task_id."""
        results: Dict[str, SolverResult] = {}
        for task in tasks:
            results[task.task_id] = self.solve(task)
        return results

    def evaluate(
        self,
        tasks: List[ArcTask],
    ) -> Dict[str, Any]:
        """Evaluate on tasks with known test outputs."""
        results = self.solve_batch(tasks)
        total = len(tasks)
        correct = 0
        for task in tasks:
            r = results[task.task_id]
            if task.test_outputs and r.predictions:
                if all(
                    p is not None and grids_equal(p, gt)
                    for p, gt in zip(r.predictions, task.test_outputs)
                ):
                    correct += 1
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
            "results": results,
        }

    @property
    def last_analysis(self) -> Optional[TaskPattern]:
        """The pattern analysis of the most recently solved task."""
        return self._last_analysis
