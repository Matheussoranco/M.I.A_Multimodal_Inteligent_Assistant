"""
Grid DSL — Domain-Specific Language for 2-D Grid Transformations
=================================================================

A comprehensive library of **pure-function** primitives for manipulating
small 2-D integer grids (≤30×30, colours 0–9).  Designed for ARC-AGI
but useful for any grid-based reasoning task.

Design principles
-----------------
* **Pure functions** — no side-effects, no state.
* **NumPy-native** — every ``Grid`` is a 2-D ``np.ndarray`` of dtype int.
* **Composable** — chain any sequence via ``compose(grid, program)``.
* **Self-describing** — each primitive is registered in ``PRIMITIVES``
  with arity, parameter schema, and human-readable description so that
  the program-synthesis engine can enumerate and search over them.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Type aliases
# ═══════════════════════════════════════════════════════════════════════

Grid = np.ndarray  # 2-D int array


@dataclass(frozen=True)
class GridObject:
    """A connected component (object) found inside a grid."""

    pixels: FrozenSet[Tuple[int, int]]
    color: int
    bbox: Tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max)

    @property
    def size(self) -> int:
        return len(self.pixels)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    def as_grid(self, bg: int = 0) -> Grid:
        """Render this object as a tight grid (bounding-box sized)."""
        r0, c0, r1, c1 = self.bbox
        g = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in self.pixels:
            g[r - r0, c - c0] = self.color
        return g


@dataclass
class PrimitiveSpec:
    """Registry entry describing a single DSL primitive."""

    name: str
    fn: Callable[..., Any]
    arity: int  # 0 = grid-only, 1+ = extra params
    param_schema: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: Tuple[str, ...] = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# 1. Grid-Geometry Primitives
# ═══════════════════════════════════════════════════════════════════════


def rotate_cw(grid: Grid) -> Grid:
    """Rotate 90° clockwise."""
    return np.rot90(grid, k=-1).copy()


def rotate_ccw(grid: Grid) -> Grid:
    """Rotate 90° counter-clockwise."""
    return np.rot90(grid, k=1).copy()


def rotate_180(grid: Grid) -> Grid:
    """Rotate 180°."""
    return np.rot90(grid, k=2).copy()


def flip_h(grid: Grid) -> Grid:
    """Mirror left ↔ right."""
    return np.fliplr(grid).copy()


def flip_v(grid: Grid) -> Grid:
    """Mirror top ↔ bottom."""
    return np.flipud(grid).copy()


def transpose(grid: Grid) -> Grid:
    """Transpose rows ↔ columns."""
    return grid.T.copy()


def crop(grid: Grid, bg: int = 0) -> Grid:
    """Remove surrounding background-colour border."""
    mask = grid != bg
    if not mask.any():
        return grid[:1, :1].copy()
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return grid[np.ix_(rows, cols)].copy()


def pad(grid: Grid, n: int = 1, color: int = 0) -> Grid:
    """Add *n* rows/columns of *color* around the grid."""
    return np.pad(grid, n, mode="constant", constant_values=color)


def tile(grid: Grid, nh: int = 2, nw: int = 2) -> Grid:
    """Tile the grid *nh × nw* times."""
    return np.tile(grid, (nh, nw))


def roll_h(grid: Grid, n: int = 1) -> Grid:
    """Circular shift columns right by *n*."""
    return np.roll(grid, n, axis=1)


def roll_v(grid: Grid, n: int = 1) -> Grid:
    """Circular shift rows down by *n*."""
    return np.roll(grid, n, axis=0)


def upscale(grid: Grid, factor: int = 2) -> Grid:
    """Scale each cell by *factor* (nearest-neighbour)."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def downscale(grid: Grid, factor: int = 2) -> Grid:
    """Downscale by *factor* (majority vote per block)."""
    h, w = grid.shape
    nh, nw = h // factor, w // factor
    if nh == 0 or nw == 0:
        return grid.copy()
    out = np.zeros((nh, nw), dtype=int)
    for r in range(nh):
        for c in range(nw):
            block = grid[r * factor : (r + 1) * factor, c * factor : (c + 1) * factor]
            vals, counts = np.unique(block, return_counts=True)
            out[r, c] = vals[np.argmax(counts)]
    return out


# ═══════════════════════════════════════════════════════════════════════
# 2. Colour Primitives
# ═══════════════════════════════════════════════════════════════════════


def replace_color(grid: Grid, old: int, new: int) -> Grid:
    """Replace every occurrence of *old* with *new*."""
    g = grid.copy()
    g[g == old] = new
    return g


def swap_colors(grid: Grid, a: int, b: int) -> Grid:
    """Swap colours *a* ↔ *b*."""
    g = grid.copy()
    mask_a = grid == a
    mask_b = grid == b
    g[mask_a] = b
    g[mask_b] = a
    return g


def palette(grid: Grid) -> Set[int]:
    """Return the set of colours present in the grid."""
    return set(int(v) for v in np.unique(grid))


def color_count(grid: Grid, color: int) -> int:
    """Count cells of a given colour."""
    return int(np.sum(grid == color))


def most_common_color(grid: Grid, exclude: Optional[Set[int]] = None) -> int:
    """Return the most-frequent colour (optionally excluding some)."""
    flat = grid.ravel()
    if exclude:
        flat = flat[~np.isin(flat, list(exclude))]
    if len(flat) == 0:
        return 0
    vals, counts = np.unique(flat, return_counts=True)
    return int(vals[np.argmax(counts)])


def least_common_color(grid: Grid, exclude: Optional[Set[int]] = None) -> int:
    """Return the least-frequent colour present."""
    flat = grid.ravel()
    if exclude:
        flat = flat[~np.isin(flat, list(exclude))]
    if len(flat) == 0:
        return 0
    vals, counts = np.unique(flat, return_counts=True)
    return int(vals[np.argmin(counts)])


def background_color(grid: Grid) -> int:
    """Heuristic: the background is the most-common colour."""
    return most_common_color(grid)


def unique_colors(grid: Grid) -> int:
    """Number of distinct colours."""
    return len(np.unique(grid))


# ═══════════════════════════════════════════════════════════════════════
# 3. Object Detection
# ═══════════════════════════════════════════════════════════════════════


def _flood_fill_component(
    grid: Grid, visited: np.ndarray, r: int, c: int, color: int
) -> FrozenSet[Tuple[int, int]]:
    """BFS flood-fill returning all connected pixels of *color*."""
    h, w = grid.shape
    queue: deque[Tuple[int, int]] = deque([(r, c)])
    visited[r, c] = True
    pixels: Set[Tuple[int, int]] = {(r, c)}
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                visited[nr, nc] = True
                pixels.add((nr, nc))
                queue.append((nr, nc))
    return frozenset(pixels)


def find_objects(grid: Grid, bg: int = 0, connectivity: int = 4) -> List[GridObject]:
    """Find all connected components (objects), ignoring background."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: List[GridObject] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color != bg and not visited[r, c]:
                pixels = _flood_fill_component(grid, visited, r, c, color)
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                bbox = (min(rows), min(cols), max(rows), max(cols))
                objects.append(GridObject(pixels=pixels, color=color, bbox=bbox))
    return objects


def find_all_objects(grid: Grid, connectivity: int = 4) -> List[GridObject]:
    """Find all connected components **including** background."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: List[GridObject] = []
    for r in range(h):
        for c in range(w):
            if not visited[r, c]:
                color = int(grid[r, c])
                pixels = _flood_fill_component(grid, visited, r, c, color)
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                bbox = (min(rows), min(cols), max(rows), max(cols))
                objects.append(GridObject(pixels=pixels, color=color, bbox=bbox))
    return objects


def object_count(grid: Grid, bg: int = 0) -> int:
    """Count non-background objects."""
    return len(find_objects(grid, bg))


def largest_object(grid: Grid, bg: int = 0) -> Optional[GridObject]:
    """Return the largest object by pixel count."""
    objs = find_objects(grid, bg)
    return max(objs, key=lambda o: o.size, default=None)


def smallest_object(grid: Grid, bg: int = 0) -> Optional[GridObject]:
    """Return the smallest object by pixel count."""
    objs = find_objects(grid, bg)
    return min(objs, key=lambda o: o.size, default=None)


def extract_object(grid: Grid, obj: GridObject, bg: int = 0) -> Grid:
    """Extract an object as a tight grid."""
    return obj.as_grid(bg)


def bounding_box(grid: Grid, color: int) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of all cells with *color*.  Returns (r0, c0, r1, c1)."""
    rows, cols = np.where(grid == color)
    if len(rows) == 0:
        return None
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


# ═══════════════════════════════════════════════════════════════════════
# 4. Object Manipulation
# ═══════════════════════════════════════════════════════════════════════


def remove_object(grid: Grid, obj: GridObject, bg: int = 0) -> Grid:
    """Erase an object (replace with background)."""
    g = grid.copy()
    for r, c in obj.pixels:
        g[r, c] = bg
    return g


def recolor_object(grid: Grid, obj: GridObject, new_color: int) -> Grid:
    """Change an object's colour."""
    g = grid.copy()
    for r, c in obj.pixels:
        g[r, c] = new_color
    return g


def move_object(grid: Grid, obj: GridObject, dr: int, dc: int, bg: int = 0) -> Grid:
    """Move an object by (dr, dc).  Wraps or clips at boundaries."""
    g = remove_object(grid, obj, bg)
    h, w = g.shape
    for r, c in obj.pixels:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            g[nr, nc] = obj.color
    return g


def copy_object(grid: Grid, obj: GridObject, dr: int, dc: int) -> Grid:
    """Copy (stamp) an object at offset (dr, dc) without removing original."""
    g = grid.copy()
    h, w = g.shape
    for r, c in obj.pixels:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            g[nr, nc] = obj.color
    return g


def gravity(grid: Grid, direction: str = "down", bg: int = 0) -> Grid:
    """Apply gravity: non-background pixels fall in *direction*."""
    g = grid.copy()
    h, w = g.shape
    if direction == "down":
        for c in range(w):
            col = [g[r, c] for r in range(h) if g[r, c] != bg]
            g[:, c] = bg
            for i, v in enumerate(col):
                g[h - len(col) + i, c] = v
    elif direction == "up":
        for c in range(w):
            col = [g[r, c] for r in range(h) if g[r, c] != bg]
            g[:, c] = bg
            for i, v in enumerate(col):
                g[i, c] = v
    elif direction == "left":
        for r in range(h):
            row = [g[r, c] for c in range(w) if g[r, c] != bg]
            g[r, :] = bg
            for i, v in enumerate(row):
                g[r, i] = v
    elif direction == "right":
        for r in range(h):
            row = [g[r, c] for c in range(w) if g[r, c] != bg]
            g[r, :] = bg
            for i, v in enumerate(row):
                g[r, w - len(row) + i] = v
    return g


def sort_objects_by_size(grid: Grid, bg: int = 0, descending: bool = True) -> List[GridObject]:
    """Return objects sorted by pixel count."""
    objs = find_objects(grid, bg)
    return sorted(objs, key=lambda o: o.size, reverse=descending)


# ═══════════════════════════════════════════════════════════════════════
# 5. Pattern & Symmetry
# ═══════════════════════════════════════════════════════════════════════


def detect_symmetry(grid: Grid) -> str:
    """Detect reflective symmetry: 'h', 'v', 'both', or 'none'."""
    h_sym = np.array_equal(grid, np.fliplr(grid))
    v_sym = np.array_equal(grid, np.flipud(grid))
    if h_sym and v_sym:
        return "both"
    if h_sym:
        return "h"
    if v_sym:
        return "v"
    return "none"


def detect_rotational_symmetry(grid: Grid) -> int:
    """Return the order of rotational symmetry (1, 2, or 4)."""
    if grid.shape[0] != grid.shape[1]:
        return 1
    if np.array_equal(grid, rotate_cw(grid)):
        return 4
    if np.array_equal(grid, rotate_180(grid)):
        return 2
    return 1


def mirror_complete_h(grid: Grid) -> Grid:
    """Mirror the left half onto the right to create horizontal symmetry."""
    g = grid.copy()
    w = g.shape[1]
    mid = w // 2
    g[:, mid:] = np.fliplr(g[:, :mid]) if w % 2 == 0 else np.fliplr(g[:, : mid + 1])[:, : w - mid]
    return g


def mirror_complete_v(grid: Grid) -> Grid:
    """Mirror the top half onto the bottom."""
    g = grid.copy()
    h = g.shape[0]
    mid = h // 2
    g[mid:, :] = np.flipud(g[:mid, :]) if h % 2 == 0 else np.flipud(g[: mid + 1, :])[:h - mid, :]
    return g


def overlay(g1: Grid, g2: Grid, bg: int = 0) -> Grid:
    """Overlay g2 on g1: non-background pixels of g2 replace g1."""
    assert g1.shape == g2.shape, "Grids must have same shape for overlay"
    result = g1.copy()
    mask = g2 != bg
    result[mask] = g2[mask]
    return result


def mask_by_color(grid: Grid, color: int) -> Grid:
    """Binary mask: 1 where colour matches, 0 elsewhere."""
    return (grid == color).astype(int)


def apply_mask(grid: Grid, mask: Grid, color: int) -> Grid:
    """Set cells to *color* wherever mask is non-zero."""
    g = grid.copy()
    g[mask != 0] = color
    return g


def flood_fill(grid: Grid, r: int, c: int, color: int) -> Grid:
    """Flood-fill from (r, c) with *color*."""
    g = grid.copy()
    h, w = g.shape
    original = g[r, c]
    if original == color:
        return g
    queue: deque[Tuple[int, int]] = deque([(r, c)])
    g[r, c] = color
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == original:
                g[nr, nc] = color
                queue.append((nr, nc))
    return g


# ═══════════════════════════════════════════════════════════════════════
# 6. Structural Operations
# ═══════════════════════════════════════════════════════════════════════


def split_h(grid: Grid, n: int = 2) -> List[Grid]:
    """Split grid into *n* horizontal strips."""
    return [s.copy() for s in np.array_split(grid, n, axis=0)]


def split_v(grid: Grid, n: int = 2) -> List[Grid]:
    """Split grid into *n* vertical strips."""
    return [s.copy() for s in np.array_split(grid, n, axis=1)]


def get_border(grid: Grid, width: int = 1) -> Grid:
    """Extract the border (outer *width* cells), interior set to bg."""
    g = np.zeros_like(grid)
    h, w_ = grid.shape
    g[:width, :] = grid[:width, :]
    g[-width:, :] = grid[-width:, :]
    g[:, :width] = grid[:, :width]
    g[:, -width:] = grid[:, -width:]
    return g


def get_interior(grid: Grid, width: int = 1) -> Grid:
    """Extract interior (without border), border set to 0."""
    g = np.zeros_like(grid)
    h, w_ = grid.shape
    if h > 2 * width and w_ > 2 * width:
        g[width:-width, width:-width] = grid[width:-width, width:-width]
    return g


def get_subgrid(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    """Extract sub-grid [r0:r1, c0:c1]."""
    return grid[r0:r1, c0:c1].copy()


def set_subgrid(grid: Grid, r0: int, c0: int, sub: Grid) -> Grid:
    """Paste *sub* into *grid* at position (r0, c0)."""
    g = grid.copy()
    sh, sw = sub.shape
    g[r0 : r0 + sh, c0 : c0 + sw] = sub
    return g


def hstack_grids(grids: Sequence[Grid], sep: int = -1, sep_color: int = 0) -> Grid:
    """Horizontally concatenate grids, optionally with a separator column."""
    if sep >= 0:
        parts: List[Grid] = []
        for i, g in enumerate(grids):
            if i > 0:
                parts.append(np.full((g.shape[0], sep), sep_color, dtype=int))
            parts.append(g)
        return np.hstack(parts)
    return np.hstack(grids)


def vstack_grids(grids: Sequence[Grid], sep: int = -1, sep_color: int = 0) -> Grid:
    """Vertically concatenate grids."""
    if sep >= 0:
        parts: List[Grid] = []
        for i, g in enumerate(grids):
            if i > 0:
                parts.append(np.full((sep, g.shape[1]), sep_color, dtype=int))
            parts.append(g)
        return np.vstack(parts)
    return np.vstack(grids)


# ═══════════════════════════════════════════════════════════════════════
# 7. Analysis Helpers
# ═══════════════════════════════════════════════════════════════════════


def grids_equal(g1: Grid, g2: Grid) -> bool:
    """Check structural equality."""
    return g1.shape == g2.shape and bool(np.array_equal(g1, g2))


def grid_diff(g1: Grid, g2: Grid) -> Grid:
    """Return a grid indicating where g1 and g2 differ (1 = diff, 0 = same)."""
    if g1.shape != g2.shape:
        h = max(g1.shape[0], g2.shape[0])
        w = max(g1.shape[1], g2.shape[1])
        return np.ones((h, w), dtype=int)
    return (g1 != g2).astype(int)


def diff_count(g1: Grid, g2: Grid) -> int:
    """Number of cells that differ between g1 and g2."""
    if g1.shape != g2.shape:
        return max(g1.size, g2.size)
    return int(np.sum(g1 != g2))


def shape_relation(inputs: List[Grid], outputs: List[Grid]) -> str:
    """Determine the size relationship between inputs and outputs."""
    if all(i.shape == o.shape for i, o in zip(inputs, outputs)):
        return "same_size"
    ratios_h = [o.shape[0] / max(i.shape[0], 1) for i, o in zip(inputs, outputs)]
    ratios_w = [o.shape[1] / max(i.shape[1], 1) for i, o in zip(inputs, outputs)]
    if len(set(ratios_h)) == 1 and len(set(ratios_w)) == 1:
        rh, rw = ratios_h[0], ratios_w[0]
        if rh == rw and rh == int(rh):
            return f"scale_{int(rh)}x"
        return f"ratio_{rh:.1f}x{rw:.1f}"
    return "variable"


def color_mapping(g_in: Grid, g_out: Grid) -> Optional[Dict[int, int]]:
    """If the transformation is a pure colour mapping, return it."""
    if g_in.shape != g_out.shape:
        return None
    mapping: Dict[int, int] = {}
    for iv, ov in zip(g_in.ravel(), g_out.ravel()):
        iv, ov = int(iv), int(ov)
        if iv in mapping:
            if mapping[iv] != ov:
                return None
        else:
            mapping[iv] = ov
    return mapping


# ═══════════════════════════════════════════════════════════════════════
# 8. Primitive Registry (for program synthesis)
# ═══════════════════════════════════════════════════════════════════════


def _build_registry() -> Dict[str, PrimitiveSpec]:
    """Build the global DSL registry used by the synthesis engine."""
    R: Dict[str, PrimitiveSpec] = {}

    def _reg(name: str, fn: Callable, arity: int = 0,
             params: Optional[Dict[str, Any]] = None,
             desc: str = "", tags: Tuple[str, ...] = ()) -> None:
        R[name] = PrimitiveSpec(
            name=name, fn=fn, arity=arity,
            param_schema=params or {}, description=desc, tags=tags,
        )

    # Geometry (arity 0 = grid-only)
    _reg("rotate_cw", rotate_cw, 0, desc="Rotate 90° CW", tags=("geometry",))
    _reg("rotate_ccw", rotate_ccw, 0, desc="Rotate 90° CCW", tags=("geometry",))
    _reg("rotate_180", rotate_180, 0, desc="Rotate 180°", tags=("geometry",))
    _reg("flip_h", flip_h, 0, desc="Mirror L↔R", tags=("geometry",))
    _reg("flip_v", flip_v, 0, desc="Mirror T↔B", tags=("geometry",))
    _reg("transpose", transpose, 0, desc="Transpose", tags=("geometry",))
    _reg("crop", crop, 0, desc="Crop to content", tags=("geometry",))

    # Parameterised geometry
    _reg("pad", pad, 1, {"n": {"type": "int", "range": [1, 5]}}, "Pad border", ("geometry",))
    _reg("tile", tile, 2, {"nh": {"type": "int", "range": [1, 4]}, "nw": {"type": "int", "range": [1, 4]}}, "Tile grid", ("geometry",))
    _reg("upscale", upscale, 1, {"factor": {"type": "int", "range": [2, 4]}}, "Upscale", ("geometry",))
    _reg("downscale", downscale, 1, {"factor": {"type": "int", "range": [2, 4]}}, "Downscale", ("geometry",))
    _reg("roll_h", roll_h, 1, {"n": {"type": "int", "range": [-5, 5]}}, "Roll horizontally", ("geometry",))
    _reg("roll_v", roll_v, 1, {"n": {"type": "int", "range": [-5, 5]}}, "Roll vertically", ("geometry",))
    _reg("gravity", gravity, 1, {"direction": {"type": "enum", "values": ["down", "up", "left", "right"]}}, "Apply gravity", ("geometry",))

    # Colour
    _reg("replace_color", replace_color, 2, {"old": {"type": "color"}, "new": {"type": "color"}}, "Replace colour", ("color",))
    _reg("swap_colors", swap_colors, 2, {"a": {"type": "color"}, "b": {"type": "color"}}, "Swap colours", ("color",))

    # Pattern
    _reg("mirror_complete_h", mirror_complete_h, 0, desc="Complete H symmetry", tags=("pattern",))
    _reg("mirror_complete_v", mirror_complete_v, 0, desc="Complete V symmetry", tags=("pattern",))
    _reg("flood_fill", flood_fill, 3, {"r": {"type": "int"}, "c": {"type": "int"}, "color": {"type": "color"}}, "Flood fill", ("pattern",))

    # Structural
    _reg("get_border", get_border, 0, desc="Extract border", tags=("structural",))
    _reg("get_interior", get_interior, 0, desc="Extract interior", tags=("structural",))

    return R


PRIMITIVES: Dict[str, PrimitiveSpec] = _build_registry()

# Zero-arity primitives (accept only a grid — no extra params)
ZERO_ARITY_NAMES: List[str] = [n for n, p in PRIMITIVES.items() if p.arity == 0]


# ═══════════════════════════════════════════════════════════════════════
# 9. Program Composition & Verification
# ═══════════════════════════════════════════════════════════════════════

# A Program is a list of (op_name, params_dict) tuples.
Program = List[Tuple[str, Dict[str, Any]]]


def compose(grid: Grid, program: Program) -> Grid:
    """Execute *program* (list of (op, params)) on *grid*."""
    for op_name, params in program:
        spec = PRIMITIVES.get(op_name)
        if spec is None:
            raise ValueError(f"Unknown primitive: {op_name}")
        grid = spec.fn(grid, **params)
    return grid


def verify_program(
    program: Program,
    examples: List[Tuple[Grid, Grid]],
) -> bool:
    """Return True iff *program* correctly maps every input → output."""
    for inp, out in examples:
        try:
            result = compose(inp.copy(), program)
            if not grids_equal(result, out):
                return False
        except Exception:
            return False
    return True


def program_score(
    program: Program,
    examples: List[Tuple[Grid, Grid]],
) -> float:
    """Score a program: fraction of examples it solves correctly."""
    if not examples:
        return 0.0
    correct = 0
    for inp, out in examples:
        try:
            result = compose(inp.copy(), program)
            if grids_equal(result, out):
                correct += 1
        except Exception:
            pass
    return correct / len(examples)


def program_distance(
    program: Program,
    examples: List[Tuple[Grid, Grid]],
) -> float:
    """Average cell-level distance (lower is better)."""
    total = 0.0
    count = 0
    for inp, out in examples:
        try:
            result = compose(inp.copy(), program)
            if result.shape == out.shape:
                total += diff_count(result, out)
            else:
                total += max(result.size, out.size)
            count += 1
        except Exception:
            total += out.size
            count += 1
    return total / max(count, 1)
