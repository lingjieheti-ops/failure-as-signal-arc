"""Smoke test: does any SINGLE generic DSL primitive solve eval tasks (depth-1 DSL reach)?
Compares against the geometry-only baseline (which got 0/120). Validates symmetry/object prims."""
import sys, os, collections, time
sys.path.insert(0, os.path.dirname(__file__))
import arc_core as ac
from dsl import PRIMS

def consistent(prim, pairs):
    for p in pairs:
        try:
            r = prim(p["input"])
        except Exception:
            return False
        if not ac.eq(r, p["output"]):
            return False
    return True

def main(split="evaluation"):
    tasks = ac.load_tasks(split)
    solved, has_rule = 0, 0
    by_prim = collections.Counter()
    t0 = time.time()
    for tid, t in sorted(tasks.items()):
        hit = None
        for name, fn in PRIMS.items():
            if name == "identity":
                continue
            if consistent(fn, t["train"]):
                hit = name
                break
        if hit:
            has_rule += 1
            # apply to test
            ok = True
            for tp in t["test"]:
                pred = PRIMS[hit](tp["input"])
                if tp["output"] is None or not ac.eq(pred, tp["output"]):
                    ok = False
            if ok:
                solved += 1
                by_prim[hit] += 1
            if tid == "0934a4d8" or hit in ("restore_occluded_region", "complete_symmetry"):
                print(f"  [{tid}] consistent single-prim = {hit}  test_solved={ok}")
    dt = time.time() - t0
    print(f"\n{split}: {len(tasks)} tasks | single-prim consistent-on-train: {has_rule} | SOLVED test: {solved} ({100*solved/len(tasks):.1f}%) | {dt:.1f}s")
    print("solved by primitive:", dict(by_prim))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation")
