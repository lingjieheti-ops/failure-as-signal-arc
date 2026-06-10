"""Depth-1 transform bank baseline for ARC-AGI-2.

Each rule factory inspects the train pairs and returns an `apply(input)->grid` callable
ONLY if the rule reproduces EVERY train output exactly (consistency-verified). The solver
collects all consistent rules, predicts on each test input, and returns the top-2 distinct
predictions as attempt_1/attempt_2 (fallback: identity). This is (a) the hard-gate valid
submission and (b) the seed of the symbolic substrate. No test outputs are ever read.
"""
from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np
from arc_core import Grid, eq, background

# ---- D4 dihedral transforms (consistency check makes liberal candidates safe) ----
D4 = {
    "identity": lambda g: g,
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "fliplr": lambda g: np.fliplr(g),
    "flipud": lambda g: np.flipud(g),
    "transpose": lambda g: g.T,
    "anti_transpose": lambda g: np.rot90(np.fliplr(g), 1),
}

def _fits(fn: Callable[[Grid], Grid], pairs) -> bool:
    for p in pairs:
        try:
            if not eq(fn(p["input"]), p["output"]):
                return False
        except Exception:
            return False
    return True

# ---- rule factories: return (name, apply_fn) if consistent on ALL train pairs, else None ----
def rf_geom(pairs):
    out = []
    for name, fn in D4.items():
        if _fits(fn, pairs):
            out.append((name, fn))
    return out

def rf_constant(pairs):
    outs = [p["output"] for p in pairs]
    if all(eq(o, outs[0]) for o in outs):
        const = outs[0].copy()
        return [("constant_output", lambda g, c=const: c)]
    return []

def rf_color_map(pairs):
    # only if every pair is shape-preserving and a consistent cellwise color map exists
    if not all(p["input"].shape == p["output"].shape for p in pairs):
        return []
    cmap = {}
    for p in pairs:
        for iv, ov in zip(p["input"].ravel(), p["output"].ravel()):
            iv, ov = int(iv), int(ov)
            if iv in cmap and cmap[iv] != ov:
                return []
            cmap[iv] = ov
    def apply(g, m=cmap):
        out = g.copy()
        for iv, ov in m.items():
            out[g == iv] = ov
        return out
    if _fits(apply, pairs):
        return [("color_map", apply)]
    return []

def rf_tile(pairs):
    # output == np.tile(input, (a,b)) with consistent integer (a,b)
    facs = set()
    for p in pairs:
        ih, iw = p["input"].shape; oh, ow = p["output"].shape
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return []
        facs.add((oh // ih, ow // iw))
    if len(facs) != 1:
        return []
    a, b = facs.pop()
    if a == 1 and b == 1:
        return []
    fn = lambda g, a=a, b=b: np.tile(g, (a, b))
    return [("tile", fn)] if _fits(fn, pairs) else []

def rf_scale(pairs):
    # each cell -> k x l block (np.kron with ones)
    facs = set()
    for p in pairs:
        ih, iw = p["input"].shape; oh, ow = p["output"].shape
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return []
        facs.add((oh // ih, ow // iw))
    if len(facs) != 1:
        return []
    k, l = facs.pop()
    if k == 1 and l == 1:
        return []
    fn = lambda g, k=k, l=l: np.kron(g, np.ones((k, l), dtype=g.dtype))
    return [("scale_up", fn)] if _fits(fn, pairs) else []

def rf_crop_content(pairs):
    def apply(g):
        bg = background(g)
        mask = g != bg
        if not mask.any():
            return g
        rs, cs = np.where(mask)
        return g[rs.min():rs.max() + 1, cs.min():cs.max() + 1]
    return [("crop_content", apply)] if _fits(apply, pairs) else []

FACTORIES = [rf_constant, rf_color_map, rf_geom, rf_tile, rf_scale, rf_crop_content]
# priority order for choosing attempt_1/2 (more specific evidence first); identity last
PRIORITY = ["constant_output", "color_map", "tile", "scale_up", "crop_content",
            "rot180", "fliplr", "flipud", "transpose", "anti_transpose",
            "rot90", "rot270", "identity"]

def solve_task(task) -> (List[dict], List[str]):
    """Return (predictions per test input, list of consistent rule names)."""
    pairs = task["train"]
    rules = []
    for f in FACTORIES:
        rules.extend(f(pairs))
    rank = {n: i for i, n in enumerate(PRIORITY)}
    rules.sort(key=lambda nr: rank.get(nr[0], 999))
    names = [n for n, _ in rules]

    preds = []
    for tp in task["test"]:
        g = tp["input"]
        cand: List[Grid] = []
        for _, fn in rules:
            try:
                pr = fn(g)
                if not any(eq(pr, c) for c in cand):
                    cand.append(pr)
            except Exception:
                pass
        if not any(eq(g, c) for c in cand):
            cand.append(g)  # identity fallback always available
        a1 = cand[0]
        a2 = cand[1] if len(cand) > 1 else cand[0]
        preds.append({"attempt_1": [[int(v) for v in r] for r in a1],
                      "attempt_2": [[int(v) for v in r] for r in a2]})
    return preds, names
