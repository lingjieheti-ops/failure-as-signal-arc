"""C3 — residual diagnostic battery: name the abstraction each task REQUIRES.

Each detector tests whether the train pairs exhibit a particular generic structure
(symmetry / periodicity / scaling / panel-logic / recolor / counting / ...), independent
of whether our solver can currently execute it. Running this over a dataset yields a
TAXONOMY: which abstraction families ARC-AGI-2 needs, and where our library's gap is.

This is the engine behind "the solver knows what it's missing": for a failed task, the
firing detector(s) name the missing primitive; adding it should close the gap (C3b).
"""
from __future__ import annotations
import sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from dsl import (symmetry_repair, symmetry_repair_extract, crop_content, crop_to_period,
                 background, _components, _bbox)

def _eq(a, b):
    return a.shape == b.shape and np.array_equal(a, b)

def _all(pairs, fn):
    try:
        return all(_eq(fn(p["input"]), p["output"]) for p in pairs)
    except Exception:
        return False

# ---- detectors: return True iff the structure holds across ALL train pairs ----
def d_constant(pairs):
    outs = [p["output"] for p in pairs]
    return len(outs) > 1 and all(_eq(o, outs[0]) for o in outs)

def d_recolor(pairs):
    if not all(p["input"].shape == p["output"].shape for p in pairs):
        return False
    m = {}
    for p in pairs:
        for a, b in zip(p["input"].ravel(), p["output"].ravel()):
            a, b = int(a), int(b)
            if a in m and m[a] != b:
                return False
            m[a] = b
    return any(k != v for k, v in m.items())

def d_symmetry_complete(pairs): return _all(pairs, symmetry_repair)
def d_symmetry_extract(pairs):  return _all(pairs, symmetry_repair_extract)
def d_periodic(pairs):          return _all(pairs, crop_to_period)
def d_crop_content(pairs):      return _all(pairs, crop_content)

def d_scale(pairs):
    facs = set()
    for p in pairs:
        ih, iw = p["input"].shape; oh, ow = p["output"].shape
        if ih and iw and oh % ih == 0 and ow % iw == 0:
            facs.add(("up", oh // ih, ow // iw))
        elif oh and ow and ih % oh == 0 and iw % ow == 0:
            facs.add(("down", ih // oh, iw // ow))
        else:
            return False
    if len(facs) != 1:
        return False
    kind, a, b = facs.pop()
    if a == 1 and b == 1:
        return False
    if kind == "up":
        return _all(pairs, lambda g, a=a, b=b: np.kron(g, np.ones((a, b), dtype=g.dtype)))
    return _all(pairs, lambda g, a=a, b=b: g[::a, ::b])

def _halves(g, axis):
    H, W = g.shape
    if axis == 1:  # vertical split into left|right
        if W % 2 == 1:  # middle column might be separator
            return g[:, :W // 2], g[:, W // 2 + 1:]
        return g[:, :W // 2], g[:, W // 2:]
    else:
        if H % 2 == 1:
            return g[:H // 2, :], g[H // 2 + 1:, :]
        return g[:H // 2, :], g[H // 2:, :]

def d_panel_logic(pairs):
    """input = two equal panels; output = AND/OR/XOR/DIFF of their non-bg masks (any fg color)."""
    for axis in (1, 0):
        for op in ("and", "or", "xor", "diff"):
            ok = True
            for p in pairs:
                try:
                    A, B = _halves(p["input"], axis)
                    if A.shape != B.shape or A.shape != p["output"].shape:
                        ok = False; break
                    bgA, bgB = background(p["input"]), background(p["input"])
                    ma, mb = (A != bgA), (B != bgB)
                    res = {"and": ma & mb, "or": ma | mb, "xor": ma ^ mb, "diff": ma & ~mb}[op]
                    out = p["output"]; obg = background(out)
                    if not np.array_equal((out != obg), res):
                        ok = False; break
                except Exception:
                    ok = False; break
            if ok:
                return True
    return False

def d_object_count(pairs):
    rels = []
    for p in pairs:
        nobj = len(_components(p["input"]))
        ncol = len(set(int(v) for v in p["input"].ravel())) - 1
        oh, ow = p["output"].shape
        feats = set()
        for nm, n in (("obj", nobj), ("col", ncol)):
            if n == oh: feats.add((nm, "H"))
            if n == ow: feats.add((nm, "W"))
            if n == oh * ow: feats.add((nm, "area"))
            if n == max(oh, ow): feats.add((nm, "max"))
        rels.append(feats)
    common = set.intersection(*rels) if rels else set()
    return len(common) > 0

def d_dedup(pairs):
    def dedup(g):
        rows = [g[0]]
        for r in g[1:]:
            if not np.array_equal(r, rows[-1]): rows.append(r)
        h = np.array(rows)
        cols = [h[:, 0]]
        for c in h.T[1:]:
            if not np.array_equal(c, cols[-1]): cols.append(c)
        return np.array(cols).T
    return _all(pairs, dedup) and not _all(pairs, lambda g: g)  # non-trivial

def d_same_shape(pairs):
    return all(p["input"].shape == p["output"].shape for p in pairs)

DETECTORS = [
    ("constant", d_constant), ("recolor", d_recolor),
    ("symmetry_complete", d_symmetry_complete), ("symmetry_extract", d_symmetry_extract),
    ("periodic", d_periodic), ("scale", d_scale), ("crop_content", d_crop_content),
    ("panel_logic", d_panel_logic), ("object_count", d_object_count), ("dedup", d_dedup),
]

def diagnose_task(task):
    fired = []
    for name, fn in DETECTORS:
        try:
            if fn(task["train"]):
                fired.append(name)
        except Exception:
            pass
    return fired

def main(split="evaluation"):
    import arc_core as ac
    tasks = ac.load_tasks(split)
    hist = collections.Counter()
    diagnosable = 0
    shape_change = 0
    for tid, t in tasks.items():
        fired = diagnose_task(t)
        if fired:
            diagnosable += 1
            for f in fired:
                hist[f] += 1
        else:
            hist["UNKNOWN"] += 1
        if not d_same_shape(t["train"]):
            shape_change += 1
    n = len(tasks)
    print(f"=== diagnostic taxonomy [{split}]: {n} tasks ===")
    print(f"diagnosable (>=1 detector fires): {diagnosable}/{n} ({100*diagnosable/n:.1f}%)")
    print(f"shape-changing tasks: {shape_change}/{n} ({100*shape_change/n:.1f}%)")
    for name, c in hist.most_common():
        print(f"  {name:18s} {c:4d}  ({100*c/n:.1f}%)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation")
