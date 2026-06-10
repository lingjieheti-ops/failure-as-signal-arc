"""Validate a submission.json against Kaggle ARC-AGI-2 format rules.

Checks: every challenge task_id present; one entry per test input; attempt_1 & attempt_2
present; each grid is a rectangular list-of-lists of ints in 0..9 with dims 1..30.
Exit 0 = valid (hard-gate format OK), exit 2 = invalid.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import arc_core as ac

def valid_grid(g):
    if not isinstance(g, list) or len(g) == 0 or not (1 <= len(g) <= 30):
        return False
    w = None
    for row in g:
        if not isinstance(row, list) or not (1 <= len(row) <= 30):
            return False
        if w is None: w = len(row)
        if len(row) != w: return False
        for v in row:
            if not isinstance(v, int) or not (0 <= v <= 9):
                return False
    return True

def main(split="evaluation", path="experiments/sub_baseline_eval.json"):
    tasks = ac.load_tasks(split)
    with open(path) as f:
        sub = json.load(f)
    errs = []
    for tid, t in tasks.items():
        if tid not in sub:
            errs.append(f"missing task {tid}"); continue
        entries = sub[tid]
        if not isinstance(entries, list) or len(entries) != len(t["test"]):
            errs.append(f"{tid}: expected {len(t['test'])} entries, got {len(entries) if isinstance(entries,list) else type(entries)}"); continue
        for i, e in enumerate(entries):
            for k in ("attempt_1", "attempt_2"):
                if k not in e:
                    errs.append(f"{tid}[{i}]: missing {k}")
                elif not valid_grid(e[k]):
                    errs.append(f"{tid}[{i}].{k}: invalid grid")
    extra = set(sub) - set(tasks)
    print(f"tasks={len(tasks)} submission_keys={len(sub)} extra_keys={len(extra)} errors={len(errs)}")
    for e in errs[:20]:
        print("  ERR", e)
    if errs:
        print("INVALID"); sys.exit(2)
    print("VALID — hard-gate format OK")

if __name__ == "__main__":
    main()
