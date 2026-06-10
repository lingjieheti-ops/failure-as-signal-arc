"""C3b — the quantified gap-closure loop (Progress evidence).

For each NAMED abstraction family F that the diagnostic battery (diagnose.py) detects, we have a
correspondingly-NAMED knowledge-free primitive `rule_F`. We measure, over a dataset:
    detector_fires[F]  = #tasks the diagnostic labels as family F
    rule_solves[F]     = #of those the named primitive actually solves (train-consistent AND test-correct)
=> "the solver diagnoses 'missing F' on N tasks, and adding the named primitive F closes M of them."
This is the interpretable, knowledge-free self-extension loop (vs SOAR/TTT black boxes).

Every rule is consistency-verified on ALL train pairs before its test prediction is trusted.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import arc_core as ac
from arc_core import eq
from dsl import symmetry_repair, symmetry_repair_extract, crop_to_period, background
from solver_baseline import rf_color_map, rf_scale, rf_constant
from diagnose import diagnose_task, _halves

def _consistent(fn, pairs):
    try:
        return all(eq(fn(p["input"]), p["output"]) for p in pairs)
    except Exception:
        return False

# ---- named rules: task -> apply_fn (train-consistent) or None ----
def rule_constant(task):
    r = rf_constant(task["train"]); return r[0][1] if r else None
def rule_recolor(task):
    r = rf_color_map(task["train"]); return r[0][1] if r else None
def rule_scale(task):
    r = rf_scale(task["train"]); return r[0][1] if r else None
def rule_periodic(task):
    return crop_to_period if _consistent(crop_to_period, task["train"]) else None
def rule_symmetry(task):
    for fn in (symmetry_repair, symmetry_repair_extract):
        if _consistent(fn, task["train"]):
            return fn
    return None
def rule_dedup(task):
    def dedup(g):
        rows = [g[0]]
        for r in g[1:]:
            if not np.array_equal(r, rows[-1]): rows.append(r)
        h = np.array(rows)
        cols = [h[:, 0]]
        for c in h.T[1:]:
            if not np.array_equal(c, cols[-1]): cols.append(c)
        return np.array(cols).T
    return dedup if (_consistent(dedup, task["train"]) and not _consistent(lambda g: g, task["train"])) else None
def rule_panel(task):
    for axis in (1, 0):
        for op in ("and", "or", "xor", "diff"):
            # infer output fg/bg color from first train pair
            o0 = task["train"][0]["output"]; obg = background(o0)
            nz = [int(c) for c in np.unique(o0) if int(c) != obg]
            ofg = nz[0] if nz else obg
            def fn(g, axis=axis, op=op, obg=obg, ofg=ofg):
                A, B = _halves(g, axis)
                if A.shape != B.shape: raise ValueError
                bg = background(g)
                ma, mb = (A != bg), (B != bg)
                m = {"and": ma & mb, "or": ma | mb, "xor": ma ^ mb, "diff": ma & ~mb}[op]
                out = np.full(A.shape, obg, dtype=g.dtype); out[m] = ofg
                return out
            if _consistent(fn, task["train"]):
                return fn
    return None

# diagnose-label -> named rule
from counting import rule_object_count
NAMED = {
    "constant": rule_constant, "recolor": rule_recolor, "scale": rule_scale,
    "periodic": rule_periodic, "symmetry_complete": rule_symmetry,
    "symmetry_extract": rule_symmetry, "dedup": rule_dedup, "panel_logic": rule_panel,
    "object_count": rule_object_count,
}

def solves(fn, task):
    if fn is None: return False
    try:
        for tp in task["test"]:
            if tp["output"] is None or not eq(fn(tp["input"]), tp["output"]):
                return False
        return True
    except Exception:
        return False

def main(split="training"):
    tasks = ac.load_tasks(split)
    fires = collections.Counter(); closes = collections.Counter()
    union = set()
    for tid, t in tasks.items():
        labs = set(diagnose_task(t))
        for lab in labs:
            fires[lab] += 1
            if lab in NAMED:
                fn = NAMED[lab](t)
                if solves(fn, t):
                    closes[lab] += 1
                    union.add(tid)
    n = len(tasks)
    print(f"=== gap-closure [{split}]: {n} tasks ===")
    print(f"{'family':18s} {'diagnosed':>9s} {'closed-by-named-primitive':>26s}")
    for lab in sorted(fires, key=lambda x: -fires[x]):
        cl = closes.get(lab, 0)
        rate = f"{100*cl/fires[lab]:.0f}%" if fires[lab] else "-"
        print(f"  {lab:18s} {fires[lab]:7d}   {cl:5d}  ({rate})")
    print(f"\nUNION solved by named-diagnosis primitives: {len(union)}/{n} ({100*len(union)/n:.1f}%)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training")
