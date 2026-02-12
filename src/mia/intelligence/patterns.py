"""
Pattern Recognition — Cross-Domain Feature Extraction & Matching
=================================================================

Provides reusable pattern-detection algorithms that work across
modalities (grids, sequences, text).  These are *code-level
intelligence structures* — deterministic algorithms, not LLM calls.

Capabilities
------------
* **Grid symmetry / periodicity detection**
* **Sequence motif discovery** (repeating sub-sequences)
* **Analogical mapping** (A:B :: C:? reasoning)
* **Abstract feature extraction** (object counts, colour histograms,
  shape descriptors, spatial relations)
* **Rule induction** from (input, output) pairs

These feed into the program synthesiser and ARC solver as
feature-extraction front-ends.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from .grid_dsl import (
    Grid,
    GridObject,
    background_color,
    color_mapping,
    crop,
    detect_rotational_symmetry,
    detect_symmetry,
    diff_count,
    find_objects,
    grids_equal,
    palette,
    shape_relation,
    unique_colors,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Grid Feature Extraction
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GridFeatures:
    """Compact feature vector describing a single grid."""

    height: int = 0
    width: int = 0
    num_colors: int = 0
    num_objects: int = 0
    bg_color: int = 0
    palette: Tuple[int, ...] = ()
    symmetry: str = "none"
    rotational_symmetry: int = 1
    color_histogram: Dict[int, int] = field(default_factory=dict)
    has_border: bool = False
    is_square: bool = False
    density: float = 0.0  # fraction of non-bg cells
    object_sizes: Tuple[int, ...] = ()
    object_colors: Tuple[int, ...] = ()
    unique_shapes: int = 0


def extract_features(grid: Grid) -> GridFeatures:
    """Extract a comprehensive feature set from a grid."""
    h, w = grid.shape
    bg = background_color(grid)
    objs = find_objects(grid, bg)
    pal = sorted(palette(grid))

    hist: Dict[int, int] = {}
    for v in np.unique(grid):
        hist[int(v)] = int(np.sum(grid == v))

    non_bg = int(np.sum(grid != bg))

    # Check for border
    has_border = False
    if h >= 3 and w >= 3:
        border_vals = set()
        border_vals.update(int(v) for v in grid[0, :])
        border_vals.update(int(v) for v in grid[-1, :])
        border_vals.update(int(v) for v in grid[:, 0])
        border_vals.update(int(v) for v in grid[:, -1])
        has_border = len(border_vals) == 1 and border_vals != {bg}

    # Count unique object shapes (normalised)
    shape_set: Set[bytes] = set()
    for obj in objs:
        sub = obj.as_grid(bg)
        cropped = crop(sub, bg)
        shape_set.add(cropped.tobytes() + bytes(cropped.shape))
    
    return GridFeatures(
        height=h,
        width=w,
        num_colors=len(pal),
        num_objects=len(objs),
        bg_color=bg,
        palette=tuple(pal),
        symmetry=detect_symmetry(grid),
        rotational_symmetry=detect_rotational_symmetry(grid),
        color_histogram=hist,
        has_border=has_border,
        is_square=(h == w),
        density=non_bg / max(h * w, 1),
        object_sizes=tuple(sorted((o.size for o in objs), reverse=True)),
        object_colors=tuple(sorted(set(o.color for o in objs))),
        unique_shapes=len(shape_set),
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. Task-Level Pattern Detection
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TaskPattern:
    """Patterns observed across all (input, output) pairs of a task."""

    size_relation: str = "unknown"
    color_mapping: Optional[Dict[int, int]] = None
    objects_change: str = "unknown"  # "same_count", "increase", "decrease", "varies"
    symmetry_preserved: bool = False
    symmetry_added: bool = False
    bg_consistent: bool = True
    common_tags: List[str] = field(default_factory=list)


def analyse_task(
    examples: List[Tuple[Grid, Grid]],
) -> TaskPattern:
    """Detect patterns in a set of (input, output) pairs."""
    inputs = [e[0] for e in examples]
    outputs = [e[1] for e in examples]

    pat = TaskPattern()

    # Size relation
    pat.size_relation = shape_relation(inputs, outputs)

    # Colour mapping
    mappings = [color_mapping(i, o) for i, o in examples]
    if all(m is not None for m in mappings):
        ref = mappings[0]
        if all(m == ref for m in mappings):
            pat.color_mapping = ref
            pat.common_tags.append("pure_color_remap")

    # Object count changes
    in_counts = [len(find_objects(g, background_color(g))) for g in inputs]
    out_counts = [len(find_objects(g, background_color(g))) for g in outputs]
    diffs = [o - i for i, o in zip(in_counts, out_counts)]
    if all(d == 0 for d in diffs):
        pat.objects_change = "same_count"
    elif all(d > 0 for d in diffs):
        pat.objects_change = "increase"
    elif all(d < 0 for d in diffs):
        pat.objects_change = "decrease"
    else:
        pat.objects_change = "varies"

    # Symmetry
    in_syms = [detect_symmetry(g) for g in inputs]
    out_syms = [detect_symmetry(g) for g in outputs]
    pat.symmetry_preserved = in_syms == out_syms
    pat.symmetry_added = all(
        os != "none" and is_ == "none"
        for is_, os in zip(in_syms, out_syms)
    )
    if pat.symmetry_added:
        pat.common_tags.append("add_symmetry")

    # Background consistency
    in_bgs = [background_color(g) for g in inputs]
    out_bgs = [background_color(g) for g in outputs]
    pat.bg_consistent = len(set(in_bgs)) == 1 and len(set(out_bgs)) == 1

    # Heuristic tags
    if pat.size_relation == "same_size":
        pat.common_tags.append("same_size")
    if pat.size_relation.startswith("scale_"):
        pat.common_tags.append("scaling")
    if all(
        grids_equal(i, o) or diff_count(i, o) <= 5
        for i, o in examples
        if i.shape == o.shape
    ):
        pat.common_tags.append("minimal_edit")

    return pat


# ═══════════════════════════════════════════════════════════════════════
# 3. Periodicity / Tiling Detection
# ═══════════════════════════════════════════════════════════════════════


def detect_periodicity(grid: Grid) -> Optional[Tuple[int, int]]:
    """Detect if the grid is a tiling of a smaller pattern.

    Returns (ph, pw) — the period — or None.
    """
    h, w = grid.shape
    for ph in range(1, h // 2 + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w // 2 + 1):
            if w % pw != 0:
                continue
            tile = grid[:ph, :pw]
            match = True
            for rr in range(0, h, ph):
                for cc in range(0, w, pw):
                    if not np.array_equal(grid[rr:rr + ph, cc:cc + pw], tile):
                        match = False
                        break
                if not match:
                    break
            if match and (ph < h or pw < w):
                return (ph, pw)
    return None


# ═══════════════════════════════════════════════════════════════════════
# 4. Sequence Motif Detection
# ═══════════════════════════════════════════════════════════════════════


def find_repeating_motif(seq: Sequence[int]) -> Optional[Tuple[int, ...]]:
    """Find the shortest repeating sub-sequence of an integer sequence."""
    n = len(seq)
    for length in range(1, n // 2 + 1):
        if n % length != 0:
            continue
        motif = tuple(seq[:length])
        if all(
            tuple(seq[i:i + length]) == motif
            for i in range(0, n, length)
        ):
            return motif
    return None


def longest_common_subsequence(a: Sequence, b: Sequence) -> int:
    """LCS length between two sequences (O(n*m))."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


# ═══════════════════════════════════════════════════════════════════════
# 5. Analogical Reasoning  (A:B :: C:?)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Analogy:
    """Represents an analogy A:B :: C:D."""

    a: Any
    b: Any
    c: Any
    d: Optional[Any] = None  # to solve for
    transform_desc: str = ""
    confidence: float = 0.0


def grid_analogy(
    a: Grid, b: Grid, c: Grid,
) -> Optional[Grid]:
    """Attempt to solve A:B :: C:?  by detecting the A→B transform
    and applying it to C.

    Tries:
    1. Pure colour remap
    2. Single DSL primitive
    3. Size change + content transform
    """
    # 1. Colour mapping
    cmap = color_mapping(a, b)
    if cmap is not None and a.shape == b.shape:
        d = c.copy()
        for old, new in cmap.items():
            d[c == old] = new
        return d

    # 2. Single-primitive scan
    from .grid_dsl import PRIMITIVES
    for name, spec in PRIMITIVES.items():
        if spec.arity > 0:
            continue
        try:
            if grids_equal(spec.fn(a), b):
                return spec.fn(c)
        except Exception:
            continue

    # 3. Delta-based: if same shape, compute pixel delta
    if a.shape == b.shape and a.shape == c.shape:
        delta_mask = (a != b)
        d = c.copy()
        d[delta_mask] = b[delta_mask]
        return d

    return None


# ═══════════════════════════════════════════════════════════════════════
# 6. Abstract Rule Induction
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class InducedRule:
    """A first-order rule induced from examples."""

    name: str
    conditions: List[str]
    action: str
    confidence: float = 1.0
    examples_supporting: int = 0


def induce_grid_rules(
    examples: List[Tuple[Grid, Grid]],
) -> List[InducedRule]:
    """Induce simple transformation rules from (input, output) pairs.

    Returns a list of human-readable rules that describe each
    transformation pattern detected.
    """
    rules: List[InducedRule] = []
    pat = analyse_task(examples)
    feats_in = [extract_features(i) for i, _ in examples]
    feats_out = [extract_features(o) for _, o in examples]

    # Rule: colour remap
    if pat.color_mapping:
        for old, new in pat.color_mapping.items():
            if old != new:
                rules.append(InducedRule(
                    name=f"recolor_{old}_to_{new}",
                    conditions=[f"pixel == {old}"],
                    action=f"set pixel to {new}",
                    examples_supporting=len(examples),
                ))

    # Rule: size change
    if pat.size_relation != "same_size":
        rules.append(InducedRule(
            name="resize",
            conditions=["grid transformation"],
            action=f"change size: {pat.size_relation}",
            examples_supporting=len(examples),
        ))

    # Rule: symmetry addition
    if pat.symmetry_added:
        target_sym = detect_symmetry(examples[0][1])
        rules.append(InducedRule(
            name="add_symmetry",
            conditions=["input has no symmetry"],
            action=f"create {target_sym} symmetry",
            examples_supporting=len(examples),
        ))

    # Rule: object count change
    if pat.objects_change == "decrease":
        rules.append(InducedRule(
            name="remove_objects",
            conditions=["multiple objects"],
            action="remove some objects (keep largest / most common)",
            examples_supporting=len(examples),
        ))
    elif pat.objects_change == "increase":
        rules.append(InducedRule(
            name="duplicate_objects",
            conditions=["objects present"],
            action="duplicate or split objects",
            examples_supporting=len(examples),
        ))

    # Rule: density change
    density_changes = [fo.density - fi.density for fi, fo in zip(feats_in, feats_out)]
    if all(d > 0.1 for d in density_changes):
        rules.append(InducedRule(
            name="fill_region",
            conditions=["sparse input"],
            action="fill empty regions",
            examples_supporting=len(examples),
        ))
    elif all(d < -0.1 for d in density_changes):
        rules.append(InducedRule(
            name="clear_region",
            conditions=["dense input"],
            action="remove content from regions",
            examples_supporting=len(examples),
        ))

    return rules


# ═══════════════════════════════════════════════════════════════════════
# 7. Object Spatial Relations
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SpatialRelation:
    """Describes a relation between two objects."""

    obj_a_idx: int
    obj_b_idx: int
    relation: str  # "above", "below", "left_of", "right_of", "overlapping", "inside", "adjacent"
    distance: float = 0.0


def compute_spatial_relations(
    objects: List[GridObject],
) -> List[SpatialRelation]:
    """Compute pairwise spatial relations between objects."""
    relations: List[SpatialRelation] = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i >= j:
                continue
            ar0, ac0, ar1, ac1 = a.bbox
            br0, bc0, br1, bc1 = b.bbox
            a_cy = (ar0 + ar1) / 2
            a_cx = (ac0 + ac1) / 2
            b_cy = (br0 + br1) / 2
            b_cx = (bc0 + bc1) / 2

            # Determine primary relation
            dy = b_cy - a_cy
            dx = b_cx - a_cx
            dist = math.sqrt(dy ** 2 + dx ** 2)

            # Check containment
            a_inside_b = br0 <= ar0 and bc0 <= ac0 and ar1 <= br1 and ac1 <= bc1
            b_inside_a = ar0 <= br0 and ac0 <= bc0 and br1 <= ar1 and bc1 <= ac1

            if a_inside_b:
                rel = "inside"
            elif b_inside_a:
                rel = "contains"
            elif abs(dy) > abs(dx):
                rel = "above" if dy > 0 else "below"
            else:
                rel = "left_of" if dx > 0 else "right_of"

            # Check adjacency
            adjacent = (
                (ar1 + 1 == br0 or br1 + 1 == ar0) and not (ac1 < bc0 or bc1 < ac0)
            ) or (
                (ac1 + 1 == bc0 or bc1 + 1 == ac0) and not (ar1 < br0 or br1 < ar0)
            )
            if adjacent and rel not in ("inside", "contains"):
                rel = "adjacent_" + rel

            relations.append(SpatialRelation(
                obj_a_idx=i,
                obj_b_idx=j,
                relation=rel,
                distance=dist,
            ))
    return relations


# ═══════════════════════════════════════════════════════════════════════
# 8. Grid Abstraction (for multi-level reasoning)
# ═══════════════════════════════════════════════════════════════════════


def abstract_grid(grid: Grid, bg: int = 0) -> Grid:
    """Produce an abstracted version: each object replaced by a single
    cell of its colour at its centroid position, on a shrunk grid."""
    objects = find_objects(grid, bg)
    if not objects:
        return grid.copy()

    # Compute centroids
    centroids = []
    for obj in objects:
        rows = [p[0] for p in obj.pixels]
        cols = [p[1] for p in obj.pixels]
        centroids.append((
            int(round(sum(rows) / len(rows))),
            int(round(sum(cols) / len(cols))),
            obj.color,
        ))

    # Build abstract grid (normalised)
    if len(centroids) == 1:
        return np.array([[centroids[0][2]]], dtype=int)

    rows = [c[0] for c in centroids]
    cols = [c[1] for c in centroids]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    abstract = np.full((h, w), bg, dtype=int)
    for r, c, color in centroids:
        abstract[r - min_r, c - min_c] = color
    return abstract
