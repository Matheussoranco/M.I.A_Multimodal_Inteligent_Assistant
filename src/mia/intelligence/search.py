"""
Search Algorithms for Reasoning & Program Synthesis
=====================================================

Provides generic search strategies over discrete state spaces.
Used by the ARC solver, program synthesiser, and cognitive kernel
for systematic exploration with bounded compute.

Algorithms
----------
* **Beam search** — maintain top-K candidates at each depth.
* **MCTS** — Monte Carlo Tree Search for reasoning trees.
* **Iterative deepening** — DFS with increasing depth limit.
* **Best-first** — A*-like search with a user-supplied heuristic.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

S = TypeVar("S")  # state type
A = TypeVar("A")  # action type


# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SearchResult(Generic[S]):
    """Outcome of a search run."""

    best_state: Optional[S] = None
    best_score: float = -math.inf
    states_explored: int = 0
    time_seconds: float = 0.0
    converged: bool = False
    all_candidates: List[Tuple[float, S]] = field(default_factory=list)


@dataclass
class MCTSNode:
    """Node in a Monte Carlo Tree."""

    state: Any
    parent: Optional["MCTSNode"] = None
    action: Any = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: List[Any] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        return self.total_value / max(self.visits, 1)

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return math.inf
        parent_visits = self.parent.visits if self.parent else 1
        return self.q_value + c * math.sqrt(math.log(parent_visits) / self.visits)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ═══════════════════════════════════════════════════════════════════════
# Beam Search
# ═══════════════════════════════════════════════════════════════════════


def beam_search(
    initial_states: Sequence[S],
    expand_fn: Callable[[S], List[S]],
    score_fn: Callable[[S], float],
    is_goal: Callable[[S], bool],
    beam_width: int = 10,
    max_depth: int = 5,
    time_limit: float = 30.0,
) -> SearchResult[S]:
    """Beam search: keep *beam_width* best candidates at each level.

    Parameters
    ----------
    initial_states : starting states
    expand_fn      : state → list of successor states
    score_fn       : state → float  (higher = better)
    is_goal        : state → bool
    beam_width     : number of candidates to keep per level
    max_depth      : maximum expansion depth
    time_limit     : wall-clock seconds budget
    """
    t0 = time.time()
    beam: List[Tuple[float, S]] = [(score_fn(s), s) for s in initial_states]
    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:beam_width]

    best_score = beam[0][0] if beam else -math.inf
    best_state = beam[0][1] if beam else None
    explored = len(beam)

    for depth in range(max_depth):
        if time.time() - t0 > time_limit:
            break

        next_beam: List[Tuple[float, S]] = []
        for _, state in beam:
            if time.time() - t0 > time_limit:
                break
            successors = expand_fn(state)
            for succ in successors:
                explored += 1
                sc = score_fn(succ)
                if is_goal(succ):
                    return SearchResult(
                        best_state=succ,
                        best_score=sc,
                        states_explored=explored,
                        time_seconds=time.time() - t0,
                        converged=True,
                    )
                next_beam.append((sc, succ))
                if sc > best_score:
                    best_score = sc
                    best_state = succ

        if not next_beam:
            break
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]

    return SearchResult(
        best_state=best_state,
        best_score=best_score,
        states_explored=explored,
        time_seconds=time.time() - t0,
        converged=False,
        all_candidates=beam[:beam_width],
    )


# ═══════════════════════════════════════════════════════════════════════
# Monte Carlo Tree Search
# ═══════════════════════════════════════════════════════════════════════


def mcts(
    root_state: Any,
    get_actions: Callable[[Any], List[Any]],
    apply_action: Callable[[Any, Any], Any],
    evaluate: Callable[[Any], float],
    is_terminal: Callable[[Any], bool],
    iterations: int = 1000,
    exploration_constant: float = 1.414,
    time_limit: float = 30.0,
) -> SearchResult:
    """Monte Carlo Tree Search.

    Parameters
    ----------
    root_state          : initial state
    get_actions         : state → list of possible actions
    apply_action        : (state, action) → new state
    evaluate            : state → float in [0, 1]  (rollout value)
    is_terminal         : state → bool
    iterations          : MCTS iterations budget
    exploration_constant: UCB1 exploration weight
    time_limit          : wall-clock seconds budget
    """
    t0 = time.time()
    root = MCTSNode(state=root_state, untried_actions=get_actions(root_state))

    best_value = -math.inf
    best_state = root_state

    for i in range(iterations):
        if time.time() - t0 > time_limit:
            break

        # 1. Selection — descend using UCB1
        node = root
        while not node.is_leaf and node.is_fully_expanded:
            node = max(node.children, key=lambda n: n.ucb1(exploration_constant))

        # 2. Expansion
        if node.untried_actions and not is_terminal(node.state):
            action = node.untried_actions.pop()
            new_state = apply_action(node.state, action)
            child = MCTSNode(
                state=new_state,
                parent=node,
                action=action,
                untried_actions=get_actions(new_state),
            )
            node.children.append(child)
            node = child

        # 3. Simulation (rollout)
        sim_state = node.state
        depth = 0
        while not is_terminal(sim_state) and depth < 20:
            actions = get_actions(sim_state)
            if not actions:
                break
            action = random.choice(actions)
            sim_state = apply_action(sim_state, action)
            depth += 1

        value = evaluate(sim_state)

        if value > best_value:
            best_value = value
            best_state = sim_state

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent  # type: ignore[assignment]

    return SearchResult(
        best_state=best_state,
        best_score=best_value,
        states_explored=i + 1 if iterations > 0 else 0,
        time_seconds=time.time() - t0,
        converged=best_value >= 1.0,
    )


# ═══════════════════════════════════════════════════════════════════════
# Iterative Deepening DFS
# ═══════════════════════════════════════════════════════════════════════


def iterative_deepening(
    initial_state: S,
    expand_fn: Callable[[S], List[S]],
    is_goal: Callable[[S], bool],
    max_depth: int = 10,
    time_limit: float = 30.0,
) -> SearchResult[S]:
    """Iterative-deepening depth-first search."""
    t0 = time.time()
    explored = 0

    for depth_limit in range(1, max_depth + 1):
        if time.time() - t0 > time_limit:
            break

        stack: List[Tuple[S, int]] = [(initial_state, 0)]
        while stack:
            if time.time() - t0 > time_limit:
                break
            state, depth = stack.pop()
            explored += 1

            if is_goal(state):
                return SearchResult(
                    best_state=state,
                    best_score=1.0,
                    states_explored=explored,
                    time_seconds=time.time() - t0,
                    converged=True,
                )

            if depth < depth_limit:
                for succ in expand_fn(state):
                    stack.append((succ, depth + 1))

    return SearchResult(
        best_state=initial_state,
        best_score=0.0,
        states_explored=explored,
        time_seconds=time.time() - t0,
        converged=False,
    )


# ═══════════════════════════════════════════════════════════════════════
# Best-First (A*-like)
# ═══════════════════════════════════════════════════════════════════════


def best_first_search(
    initial_state: S,
    expand_fn: Callable[[S], List[S]],
    score_fn: Callable[[S], float],
    is_goal: Callable[[S], bool],
    max_states: int = 10_000,
    time_limit: float = 30.0,
) -> SearchResult[S]:
    """Best-first search ordered by *score_fn* (higher = explore first)."""
    import heapq

    t0 = time.time()
    # Use negative score so heapq (min-heap) explores highest scores first
    heap: List[Tuple[float, int, S]] = [(-score_fn(initial_state), 0, initial_state)]
    explored = 0
    best_score = -math.inf
    best_state = initial_state
    counter = 1  # tiebreaker

    while heap and explored < max_states:
        if time.time() - t0 > time_limit:
            break

        neg_sc, _, state = heapq.heappop(heap)
        sc = -neg_sc
        explored += 1

        if sc > best_score:
            best_score = sc
            best_state = state

        if is_goal(state):
            return SearchResult(
                best_state=state,
                best_score=sc,
                states_explored=explored,
                time_seconds=time.time() - t0,
                converged=True,
            )

        for succ in expand_fn(state):
            succ_sc = score_fn(succ)
            heapq.heappush(heap, (-succ_sc, counter, succ))
            counter += 1

    return SearchResult(
        best_state=best_state,
        best_score=best_score,
        states_explored=explored,
        time_seconds=time.time() - t0,
        converged=False,
    )
