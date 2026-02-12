"""
Algorithmic Reasoning Engine for M.I.A
=======================================

Implements *real* computational reasoning that does NOT depend on
LLM prompt-forwarding.  Every solver here is a deterministic
algorithm that can run without any language model.

Capabilities
------------
- Constraint propagation (AC-3)
- Boolean satisfiability (DPLL-style)
- Symbolic equation solver
- Analogical / proportional reasoning
- Logical-inference engine (forward + backward chaining)
- Graph / path reasoning (BFS/Dijkstra for shortest path)
- Numerical optimisation (Nelder-Mead simplex)
"""

from __future__ import annotations

import itertools
import logging
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Deque, Dict, FrozenSet, List, Optional, Set,
    Sequence, Tuple, Union,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Constraint Satisfaction Problem (CSP) solver — AC-3 + backtracking
# ═══════════════════════════════════════════════════════════════════════════

Constraint = Callable[..., bool]


@dataclass
class CSP:
    """A generic Constraint Satisfaction Problem."""

    variables: List[str]
    domains: Dict[str, List[Any]]
    constraints: List[Tuple[Tuple[str, ...], Constraint]]

    def copy_domains(self) -> Dict[str, List[Any]]:
        return {v: list(d) for v, d in self.domains.items()}


def ac3(csp: CSP, domains: Optional[Dict[str, List[Any]]] = None) -> Optional[Dict[str, List[Any]]]:
    """Arc-consistency 3 — prune domain values that violate binary constraints."""
    domains = domains if domains is not None else csp.copy_domains()
    queue: Deque[Tuple[str, str, Constraint]] = deque()

    for scope, check in csp.constraints:
        if len(scope) == 2:
            xi, xj = scope
            queue.append((xi, xj, check))
            queue.append((xj, xi, check))

    while queue:
        xi, xj, check = queue.popleft()
        if _revise(domains, xi, xj, check):
            if not domains[xi]:
                return None  # domain wipe-out → no solution
            for scope, c in csp.constraints:
                if len(scope) == 2 and xi in scope:
                    xk = scope[0] if scope[1] == xi else scope[1]
                    if xk != xj:
                        queue.append((xk, xi, c))
    return domains


def _revise(
    domains: Dict[str, List[Any]], xi: str, xj: str, check: Constraint,
) -> bool:
    revised = False
    for val_i in list(domains[xi]):
        if not any(check(val_i, val_j) for val_j in domains[xj]):
            domains[xi].remove(val_i)
            revised = True
    return revised


def solve_csp(
    csp: CSP,
    *,
    use_ac3: bool = True,
    max_solutions: int = 1,
) -> List[Dict[str, Any]]:
    """Solve a CSP using AC-3 + backtracking with MRV heuristic."""
    domains = csp.copy_domains()
    if use_ac3:
        domains_pruned = ac3(csp, domains)
        if domains_pruned is None:
            return []
        domains = domains_pruned

    solutions: List[Dict[str, Any]] = []
    _backtrack(csp, {}, domains, solutions, max_solutions)
    return solutions


def _backtrack(
    csp: CSP,
    assignment: Dict[str, Any],
    domains: Dict[str, List[Any]],
    solutions: List[Dict[str, Any]],
    max_solutions: int,
) -> bool:
    if len(assignment) == len(csp.variables):
        solutions.append(dict(assignment))
        return len(solutions) >= max_solutions

    # MRV heuristic — pick the variable with the fewest remaining values
    unassigned = [v for v in csp.variables if v not in assignment]
    var = min(unassigned, key=lambda v: len(domains[v]))

    for value in domains[var]:
        assignment[var] = value
        if _is_consistent(csp, assignment):
            # Forward-check: copy domains and prune
            new_domains = {v: list(d) for v, d in domains.items()}
            new_domains[var] = [value]
            pruned = ac3(csp, new_domains)
            if pruned is not None:
                if _backtrack(csp, assignment, pruned, solutions, max_solutions):
                    del assignment[var]
                    return True
        del assignment[var]
    return False


def _is_consistent(csp: CSP, assignment: Dict[str, Any]) -> bool:
    for scope, check in csp.constraints:
        vals = tuple(assignment.get(v) for v in scope)
        if None in vals:
            continue  # not all variables assigned yet
        if not check(*vals):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 2. DPLL-based Boolean Satisfiability
# ═══════════════════════════════════════════════════════════════════════════

Literal = int  # positive = variable, negative = negation
Clause = FrozenSet[Literal]


def dpll(clauses: List[Clause], variables: Set[int]) -> Optional[Dict[int, bool]]:
    """DPLL satisfiability solver with unit propagation and pure-literal."""
    assignment: Dict[int, bool] = {}
    return _dpll_rec(set(clauses), variables, assignment)


def _dpll_rec(
    clauses: Set[Clause],
    variables: Set[int],
    assignment: Dict[int, bool],
) -> Optional[Dict[int, bool]]:
    # Unit propagation
    changed = True
    while changed:
        changed = False
        for clause in list(clauses):
            if len(clause) == 1:
                lit = next(iter(clause))
                var = abs(lit)
                val = lit > 0
                assignment[var] = val
                variables = variables - {var}
                clauses = _propagate(clauses, lit)
                changed = True
                break
        if frozenset() in clauses:
            return None  # conflict

    if not clauses:
        # All clauses satisfied — fill remaining variables
        for v in variables:
            assignment.setdefault(v, True)
        return assignment

    # Pure literal elimination
    all_lits = {lit for clause in clauses for lit in clause}
    for var in list(variables):
        if var in all_lits and -var not in all_lits:
            assignment[var] = True
            variables = variables - {var}
            clauses = _propagate(clauses, var)
        elif -var in all_lits and var not in all_lits:
            assignment[var] = False
            variables = variables - {var}
            clauses = _propagate(clauses, -var)

    if not clauses:
        for v in variables:
            assignment.setdefault(v, True)
        return assignment
    if frozenset() in clauses:
        return None

    # Branch on a variable
    var = next(iter(variables))
    for val in (True, False):
        lit = var if val else -var
        new_clauses = _propagate(clauses, lit)
        if frozenset() not in new_clauses:
            result = _dpll_rec(
                new_clauses, variables - {var}, {**assignment, var: val},
            )
            if result is not None:
                return result
    return None


def _propagate(clauses: Set[Clause], lit: Literal) -> Set[Clause]:
    """Propagate a literal assignment through the clause set."""
    new: Set[Clause] = set()
    for clause in clauses:
        if lit in clause:
            continue  # clause satisfied
        reduced = frozenset(l for l in clause if l != -lit)
        new.add(reduced)
    return new


# ═══════════════════════════════════════════════════════════════════════════
# 3. Symbolic equation solver (linear systems)
# ═══════════════════════════════════════════════════════════════════════════


def solve_linear_system(
    A: List[List[float]], b: List[float],
) -> Optional[List[float]]:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Returns None if the system is singular.
    """
    n = len(b)
    # Augmented matrix
    M = [row[:] + [bi] for row, bi in zip(A, b)]

    for col in range(n):
        # Partial pivot
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]

        if abs(M[col][col]) < 1e-12:
            return None  # singular

        # Eliminate below
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back-substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


# ═══════════════════════════════════════════════════════════════════════════
# 4. Analogical / proportional reasoning
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Analogy:
    """Represents A : B :: C : ?"""
    a: Any
    b: Any
    c: Any
    relation: Optional[str] = None  # description of A→B transform


def solve_numeric_analogy(a: float, b: float, c: float) -> List[float]:
    """Propose candidate answers for  a : b :: c : ?

    Tries additive, multiplicative, power, and logarithmic relations.
    """
    candidates: List[float] = []

    # Additive: b = a + d  →  ? = c + d
    d = b - a
    candidates.append(c + d)

    # Multiplicative: b = a * r  →  ? = c * r
    if a != 0:
        r = b / a
        candidates.append(c * r)

    # Power: b = a^p  →  ? = c^p
    if a > 0 and b > 0:
        try:
            p = math.log(b) / math.log(a) if a != 1 else 1
            candidates.append(c ** p)
        except (ValueError, ZeroDivisionError):
            pass

    # Square relation: b = a^2  →  ? = c^2
    if abs(b - a * a) < 1e-9:
        candidates.append(c * c)

    # Reciprocal: b = 1/a  →  ? = 1/c
    if a != 0 and abs(b - 1.0 / a) < 1e-9 and c != 0:
        candidates.append(1.0 / c)

    return list(set(round(v, 10) for v in candidates if math.isfinite(v)))


def solve_sequence_analogy(seq: List[float]) -> List[float]:
    """Given a number sequence, predict the next 1-3 values.

    Tries constant difference, constant ratio, second-order differences,
    and Fibonacci-like patterns.
    """
    if len(seq) < 2:
        return []

    predictions: List[float] = []

    # Constant difference (arithmetic)
    diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    if len(set(round(d, 10) for d in diffs)) == 1:
        d = diffs[0]
        predictions = [seq[-1] + d * (i + 1) for i in range(3)]
        return predictions

    # Constant ratio (geometric)
    if all(seq[i] != 0 for i in range(len(seq) - 1)):
        ratios = [seq[i + 1] / seq[i] for i in range(len(seq) - 1)]
        if len(set(round(r, 10) for r in ratios)) == 1:
            r = ratios[0]
            val = seq[-1]
            for _ in range(3):
                val *= r
                predictions.append(val)
            return predictions

    # Second-order differences (quadratic)
    if len(diffs) >= 2:
        diffs2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        if len(set(round(d, 10) for d in diffs2)) == 1:
            d2 = diffs2[0]
            last_diff = diffs[-1]
            val = seq[-1]
            for _ in range(3):
                last_diff += d2
                val += last_diff
                predictions.append(val)
            return predictions

    # Fibonacci-like (a[n] = a[n-1] + a[n-2])
    if len(seq) >= 3:
        is_fib = all(
            abs(seq[i] - seq[i - 1] - seq[i - 2]) < 1e-9
            for i in range(2, len(seq))
        )
        if is_fib:
            a, b = seq[-2], seq[-1]
            for _ in range(3):
                a, b = b, a + b
                predictions.append(b)
            return predictions

    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# 5. Logical-inference engine (forward / backward chaining)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Rule:
    """A propositional horn-clause rule:  premises → conclusion."""
    premises: FrozenSet[str]
    conclusion: str
    label: str = ""


class LogicEngine:
    """Forward- and backward-chaining inference over horn clauses."""

    def __init__(self) -> None:
        self.facts: Set[str] = set()
        self.rules: List[Rule] = []

    def tell(self, fact: str) -> None:
        self.facts.add(fact)

    def add_rule(self, premises: Sequence[str], conclusion: str, label: str = "") -> None:
        self.rules.append(Rule(frozenset(premises), conclusion, label))

    # ── Forward chaining ────────────────────────────────────────────

    def forward_chain(self, max_rounds: int = 100) -> Set[str]:
        """Derive all reachable facts using forward chaining."""
        inferred = set(self.facts)
        for _ in range(max_rounds):
            new_facts: Set[str] = set()
            for rule in self.rules:
                if rule.premises.issubset(inferred) and rule.conclusion not in inferred:
                    new_facts.add(rule.conclusion)
            if not new_facts:
                break
            inferred.update(new_facts)
        return inferred

    # ── Backward chaining ───────────────────────────────────────────

    def backward_chain(self, goal: str) -> bool:
        """Query whether *goal* can be derived."""
        return self._bc(goal, set())

    def _bc(self, goal: str, visited: Set[str]) -> bool:
        if goal in self.facts:
            return True
        if goal in visited:
            return False  # avoid cycles
        visited.add(goal)
        for rule in self.rules:
            if rule.conclusion == goal:
                if all(self._bc(p, visited) for p in rule.premises):
                    self.facts.add(goal)  # cache derived fact
                    return True
        return False

    def explain(self, goal: str) -> List[str]:
        """Return a proof trace for *goal*."""
        trace: List[str] = []
        self._explain(goal, set(), trace)
        return trace

    def _explain(self, goal: str, visited: Set[str], trace: List[str]) -> bool:
        if goal in self.facts:
            trace.append(f"  KNOWN: {goal}")
            return True
        if goal in visited:
            return False
        visited.add(goal)
        for rule in self.rules:
            if rule.conclusion == goal:
                if all(self._explain(p, visited, trace) for p in rule.premises):
                    label = f" ({rule.label})" if rule.label else ""
                    trace.append(
                        f"  DERIVE: {' ∧ '.join(rule.premises)} → {goal}{label}"
                    )
                    return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 6. Graph reasoning (shortest path, connected components, etc.)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Graph:
    """A weighted directed graph."""
    adjacency: Dict[str, List[Tuple[str, float]]] = field(default_factory=lambda: defaultdict(list))

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        self.adjacency[u].append((v, weight))

    def add_undirected(self, u: str, v: str, weight: float = 1.0) -> None:
        self.add_edge(u, v, weight)
        self.add_edge(v, u, weight)

    @property
    def nodes(self) -> Set[str]:
        ns: Set[str] = set()
        for u, edges in self.adjacency.items():
            ns.add(u)
            for v, _ in edges:
                ns.add(v)
        return ns


def dijkstra(graph: Graph, start: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """Dijkstra's shortest-path from *start* to all reachable nodes."""
    import heapq

    dist: Dict[str, float] = {start: 0.0}
    prev: Dict[str, Optional[str]] = {start: None}
    pq: List[Tuple[float, str]] = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph.adjacency.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def shortest_path(graph: Graph, start: str, end: str) -> Optional[List[str]]:
    """Return the shortest-weight path from *start* to *end*, or None."""
    dist, prev = dijkstra(graph, start)
    if end not in dist:
        return None
    path: List[str] = []
    node: Optional[str] = end
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path


def connected_components(graph: Graph) -> List[Set[str]]:
    """Find connected components (treats directed edges as undirected)."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    for u, edges in graph.adjacency.items():
        for v, _ in edges:
            adj[u].add(v)
            adj[v].add(u)

    visited: Set[str] = set()
    components: List[Set[str]] = []

    for node in graph.nodes:
        if node in visited:
            continue
        comp: Set[str] = set()
        queue: Deque[str] = deque([node])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in adj.get(n, set()):
                if nb not in visited:
                    queue.append(nb)
        components.append(comp)

    return components


# ═══════════════════════════════════════════════════════════════════════════
# 7. Nelder-Mead simplex optimisation (gradient-free)
# ═══════════════════════════════════════════════════════════════════════════


def nelder_mead(
    f: Callable[[List[float]], float],
    x0: List[float],
    *,
    tol: float = 1e-8,
    max_iter: int = 2000,
    initial_step: float = 0.5,
) -> Tuple[List[float], float]:
    """Minimise *f* starting from *x0* using the Nelder-Mead simplex method.

    Returns (best_point, best_value).
    """
    n = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    # Build initial simplex
    simplex: List[List[float]] = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] += initial_step
        simplex.append(point)

    values = [f(p) for p in simplex]

    for _ in range(max_iter):
        # Order
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        # Convergence check
        if max(abs(values[i] - values[0]) for i in range(1, n + 1)) < tol:
            break

        # Centroid (excluding worst)
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

        worst = simplex[-1]
        worst_val = values[-1]

        # Reflection
        xr = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        fr = f(xr)

        if values[0] <= fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        elif fr < values[0]:
            # Expansion
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        else:
            # Contraction
            xc = [centroid[j] + rho * (worst[j] - centroid[j]) for j in range(n)]
            fc = f(xc)
            if fc < worst_val:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                best = simplex[0]
                for i in range(1, n + 1):
                    simplex[i] = [
                        best[j] + sigma * (simplex[i][j] - best[j])
                        for j in range(n)
                    ]
                    values[i] = f(simplex[i])

    best_idx = min(range(n + 1), key=lambda i: values[i])
    return simplex[best_idx], values[best_idx]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Unified reasoning dispatcher
# ═══════════════════════════════════════════════════════════════════════════


class AlgorithmicReasoner:
    """Facade that exposes all algorithmic reasoning primitives.

    Unlike the LLM-based reasoners in ``reasoning.__init__``, every
    method here is a deterministic algorithm with no LLM dependency.
    """

    def __init__(self) -> None:
        self.logic = LogicEngine()

    # ── Constraint satisfaction ──────────────────────────────────

    @staticmethod
    def solve_csp(
        variables: List[str],
        domains: Dict[str, List[Any]],
        constraints: List[Tuple[Tuple[str, ...], Constraint]],
        *,
        max_solutions: int = 1,
    ) -> List[Dict[str, Any]]:
        problem = CSP(variables, domains, constraints)
        return solve_csp(problem, max_solutions=max_solutions)

    # ── SAT ──────────────────────────────────────────────────────

    @staticmethod
    def solve_sat(
        clauses: List[List[int]],
    ) -> Optional[Dict[int, bool]]:
        frozen = [frozenset(c) for c in clauses]
        variables = {abs(l) for c in clauses for l in c}
        return dpll(list(frozen), variables)

    # ── Linear algebra ───────────────────────────────────────────

    @staticmethod
    def solve_linear(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
        return solve_linear_system(A, b)

    # ── Analogy ──────────────────────────────────────────────────

    @staticmethod
    def numeric_analogy(a: float, b: float, c: float) -> List[float]:
        return solve_numeric_analogy(a, b, c)

    @staticmethod
    def sequence_predict(seq: List[float]) -> List[float]:
        return solve_sequence_analogy(seq)

    # ── Logic ────────────────────────────────────────────────────

    def tell_fact(self, fact: str) -> None:
        self.logic.tell(fact)

    def add_rule(self, premises: Sequence[str], conclusion: str, label: str = "") -> None:
        self.logic.add_rule(premises, conclusion, label)

    def ask(self, goal: str) -> bool:
        return self.logic.backward_chain(goal)

    def derive_all(self) -> Set[str]:
        return self.logic.forward_chain()

    def explain(self, goal: str) -> List[str]:
        return self.logic.explain(goal)

    # ── Graph ────────────────────────────────────────────────────

    @staticmethod
    def shortest_path(
        edges: List[Tuple[str, str, float]],
        start: str,
        end: str,
    ) -> Optional[List[str]]:
        g = Graph()
        for u, v, w in edges:
            g.add_edge(u, v, w)
        return shortest_path(g, start, end)

    # ── Optimisation ─────────────────────────────────────────────

    @staticmethod
    def minimise(
        f: Callable[[List[float]], float],
        x0: List[float],
        **kwargs: Any,
    ) -> Tuple[List[float], float]:
        return nelder_mead(f, x0, **kwargs)
