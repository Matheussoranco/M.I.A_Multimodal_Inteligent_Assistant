"""
M.I.A Intelligence Module
==========================

Code-level intelligence structures that provide genuine reasoning
capabilities beyond LLM prompt forwarding.

Sub-modules:
- ``grid_dsl``          Pure-function grid primitives (ARC-AGI foundation)
- ``arc_solver``        Multi-strategy ARC-AGI solver
- ``program_synthesis`` Example-guided program search
- ``search``            Beam search, MCTS, iterative deepening
- ``hypothesis``        Hypothesis generation, testing, ranking
- ``patterns``          Pattern recognition across domains
"""

from __future__ import annotations

__all__ = [
    "get_arc_solver",
    "get_synthesiser",
    "get_hypothesis_tester",
]

# Convenience imports for quick access
def get_arc_solver():
    """Lazy factory for ArcSolver."""
    from .arc_solver import ArcSolver
    return ArcSolver()

def get_synthesiser():
    """Lazy factory for ProgramSynthesiser."""
    from .program_synthesis import ProgramSynthesiser
    return ProgramSynthesiser()

def get_hypothesis_tester():
    """Lazy factory for HypothesisTester."""
    from .hypothesis import HypothesisTester
    return HypothesisTester()
