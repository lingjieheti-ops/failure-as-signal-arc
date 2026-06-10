"""Generic COUNTING/CARDINALITY rule families — the gap-closure loop's second iteration.

The diagnostic flagged object_count as the largest unclosed family (31 training tasks, 0% closed).
These are knowledge-free, domain-generic counting rules (count things; emit a grid whose size/content
encodes the count; or select an object by a cardinality criterion). Factory-style: each returns an
apply(grid)->grid ONLY if it reproduces EVERY train pair (so liberal hypothesis spaces stay safe).
"""
from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np
from scipy import ndimage
from arc_core import eq
from dsl import background, _bbox

# ---------- helpers ----------
def _comps(g, connectivity=2, same_color=False):
    """Connected components of non-background cells. If same_color, split by color."""
    bg = background(g)
    struct = ndimage.generate_binary_structure(2, connectivity)
    masks = []
    if same_color:
        for c in np.unique(g):
            if int(c) == bg:
                continue
            lab, n = ndimage.label(g == c, structure=struct)
            masks.extend([(lab == i) for i in range(1, n + 1)])
    else:
        lab, n = ndimage.label(g != bg, structure=struct)
        masks = [(lab == i) for i in range(1, n + 1)]
    return masks

# count sources: name -> fn(grid)->int
COUNT_SOURCES = {
    "n_obj8": lambda g: len(_comps(g, 2, False)),
    "n_obj4": lambda g: len(_comps(g, 1, False)),
    "n_obj8_bycolor": lambda g: len(_comps(g, 2, True)),
    "n_obj4_bycolor": lambda g: len(_comps(g, 1, True)),
    "n_fg_colors": lambda g: len([c for c in np.unique(g) if int(c) != background(g)]),
    "max_color_count": lambda g: int(max([(g == c).sum() for c in np.unique(g) if int(c) != background(g)], default=0)),
    "largest_obj_cells": lambda g: int(max([m.sum() for m in _comps(g)], default=0)),
}
# shapes: name -> fn(n)->(h,w)
COUNT_SHAPES = {"row": lambda n: (1, n), "col": lambda n: (n, 1), "square": lambda n: (n, n)}

def _fill_colors(g):
    """Candidate fill colors, in priority: majority fg, minority fg, largest-object color, each fg color."""
    bg = background(g)
    fg = [(int(c), int((g == c).sum())) for c in np.unique(g) if int(c) != bg]
    out = []
    if fg:
        out.append(max(fg, key=lambda t: t[1])[0])          # majority fg
        out.append(min(fg, key=lambda t: t[1])[0])          # minority fg
        comps = _comps(g)
        if comps:
            m = max(comps, key=lambda x: int(x.sum()))
            vals, cnts = np.unique(g[m], return_counts=True)
            out.append(int(vals[int(np.argmax(cnts))]))     # largest-object color
        out.extend(c for c, _ in fg)
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq

# ---------- family 1: count -> constant grid ----------
def rf_count_grid(pairs):
    """output = (h,w)=shape(count) grid filled with a color chosen by a consistent strategy."""
    for sname, sfn in COUNT_SOURCES.items():
        for shname, shfn in COUNT_SHAPES.items():
            ok_shape = True
            for p in pairs:
                try:
                    n = sfn(p["input"])
                except Exception:
                    ok_shape = False; break
                if n <= 0 or n > 30 or p["output"].shape != shfn(n):
                    ok_shape = False; break
            if not ok_shape:
                continue
            # fill strategy: (a) same color-rank rule across pairs, (b) one constant color
            strategies = []
            for rank in range(4):  # index into _fill_colors priority list
                strategies.append(("rank%d" % rank, lambda g, r=rank: (_fill_colors(g) + [background(g)] * 4)[r]))
            const_cols = set(int(v) for p in pairs for v in np.unique(p["output"]))
            if len(const_cols) == 1:
                cc = const_cols.pop()
                strategies.append(("const", lambda g, c=cc: c))
            for stname, stfn in strategies:
                def apply(g, sfn=sfn, shfn=shfn, stfn=stfn):
                    n = sfn(g)
                    n = max(1, min(int(n), 30))
                    return np.full(shfn(n), stfn(g), dtype=g.dtype)
                if all(eq(apply(p["input"]), p["output"]) for p in pairs):
                    return [(f"count_grid[{sname},{shname},{stname}]", apply)]
    return []

# ---------- family 2: select object by cardinality ----------
def _norm_shape(mask, grid):
    bb = _bbox(mask)
    r0, r1, c0, c1 = bb
    sub = np.where(mask[r0:r1+1, c0:c1+1], grid[r0:r1+1, c0:c1+1], -1)
    return sub

def _crop(mask, grid):
    r0, r1, c0, c1 = _bbox(mask)
    return grid[r0:r1+1, c0:c1+1].copy()

def rf_select_by_count(pairs):
    """output = crop of the object selected by a cardinality criterion (unique shape / modal shape /
    unique color / most-frequent color / max|min cell count), tried with both connectivities & color-split."""
    CRITS = ["unique_shape", "modal_shape", "unique_color", "modal_color", "max_cells", "min_cells"]
    for conn in (2, 1):
        for bycol in (True, False):
            for crit in CRITS:
                def apply(g, conn=conn, bycol=bycol, crit=crit):
                    comps = _comps(g, conn, bycol)
                    if not comps:
                        return g
                    if crit in ("unique_shape", "modal_shape"):
                        keys = []
                        for m in comps:
                            s = _norm_shape(m, g)
                            keys.append((s.shape, s.tobytes()))
                        from collections import Counter
                        cnt = Counter(keys)
                        if crit == "unique_shape":
                            want = [i for i, k in enumerate(keys) if cnt[k] == 1]
                        else:
                            mode = cnt.most_common(1)[0][0]
                            want = [i for i, k in enumerate(keys) if k == mode]
                        if len(want) != 1:
                            return g
                        return _crop(comps[want[0]], g)
                    if crit in ("unique_color", "modal_color"):
                        cols = []
                        for m in comps:
                            vals, cnts = np.unique(g[m], return_counts=True)
                            cols.append(int(vals[int(np.argmax(cnts))]))
                        from collections import Counter
                        cnt = Counter(cols)
                        if crit == "unique_color":
                            want = [i for i, c in enumerate(cols) if cnt[c] == 1]
                            if len(want) != 1:
                                return g
                            return _crop(comps[want[0]], g)
                        mode = cnt.most_common(1)[0][0]
                        want = [i for i, c in enumerate(cols) if c == mode]
                        if not want:
                            return g
                        m = max((comps[i] for i in want), key=lambda x: int(x.sum()))
                        return _crop(m, g)
                    sizes = [int(m.sum()) for m in comps]
                    i = int(np.argmax(sizes)) if crit == "max_cells" else int(np.argmin(sizes))
                    return _crop(comps[i], g)
                try:
                    if all(eq(apply(p["input"]), p["output"]) for p in pairs):
                        return [(f"select[{crit},conn{conn},{'col' if bycol else 'all'}]", apply)]
                except Exception:
                    continue
    return []

# ---------- family 3: learned count -> color code ----------
def rf_count_to_color_map(pairs):
    """output = 1x1 grid whose color is a per-task LEARNED function of a count (like color_map, but over
    cardinalities). Also tries symmetry/shape-category sources. Knowledge-free: mapping induced from train."""
    extra_sources = dict(COUNT_SOURCES)
    extra_sources["n_cells_fg"] = lambda g: int((g != background(g)).sum())
    extra_sources["sym_class"] = lambda g: (int(np.array_equal(g, np.fliplr(g))) +
                                            2 * int(np.array_equal(g, np.flipud(g))) +
                                            4 * int(g.shape[0] == g.shape[1] and np.array_equal(g, g.T)))
    if not all(p["output"].size == 1 for p in pairs):
        return []
    for sname, sfn in extra_sources.items():
        mapping = {}
        ok = True
        for p in pairs:
            try:
                k = sfn(p["input"])
            except Exception:
                ok = False; break
            v = int(p["output"].ravel()[0])
            if k in mapping and mapping[k] != v:
                ok = False; break
            mapping[k] = v
        if not ok or len(mapping) < 2:   # require a non-trivial learned map
            continue
        def apply(g, sfn=sfn, m=dict(mapping)):
            k = sfn(g)
            if k not in m:
                raise ValueError("count outside learned map")
            return np.array([[m[k]]], dtype=g.dtype)
        if all(eq(apply(p["input"]), p["output"]) for p in pairs):
            return [(f"count_to_color[{sname}]", apply)]
    return []

# ---------- family 4 (iteration 3, training-derived): ranked count histogram ----------
def rf_count_ranking(pairs):
    """output = bar chart of fg colors ranked by count (objects or cells), one row per color, bar length =
    count, aligned left/right, sorted desc/asc. Derived from TRAINING task family (e.g. 2753e76c)."""
    def bars(g, src, align, desc):
        bg = background(g)
        if src == "cells":
            counts = [(int(c), int((g == c).sum())) for c in np.unique(g) if int(c) != bg]
        else:
            counts = []
            struct = ndimage.generate_binary_structure(2, 2)
            for c in np.unique(g):
                if int(c) == bg:
                    continue
                _, n = ndimage.label(g == c, structure=struct)
                counts.append((int(c), int(n)))
        counts = [t for t in counts if t[1] > 0]
        if not counts:
            raise ValueError
        counts.sort(key=lambda t: (-t[1], t[0]) if desc else (t[1], t[0]))
        W = max(n for _, n in counts)
        if W > 30 or len(counts) > 30:
            raise ValueError
        out = np.zeros((len(counts), W), dtype=np.int8)
        for i, (c, n) in enumerate(counts):
            if align == "left":
                out[i, :n] = c
            else:
                out[i, W - n:] = c
        return out
    for src in ("objects", "cells"):
        for align in ("right", "left"):
            for desc in (True, False):
                def apply(g, src=src, align=align, desc=desc):
                    return bars(g, src, align, desc)
                try:
                    if all(eq(apply(p["input"]), p["output"]) for p in pairs):
                        return [(f"count_ranking[{src},{align},{'desc' if desc else 'asc'}]", apply)]
                except Exception:
                    continue
    return []

def rule_object_count(task):
    """Named rule for the 'object_count' diagnosed family. Returns apply fn or None."""
    for rf in (rf_count_grid, rf_select_by_count, rf_count_to_color_map, rf_count_ranking):
        r = rf(task["train"])
        if r:
            return r[0][1]
    return None
