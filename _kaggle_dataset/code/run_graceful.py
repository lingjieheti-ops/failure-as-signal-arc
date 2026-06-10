"""C2 fixed — conservative anytime policy: deviate from copy-input ONLY when a program beats
copying on TRAIN (a legitimate, no-test-peek decision). This is what 'graceful degradation' should
mean: never do worse than leaving the grid alone unless you have train evidence to. Re-measures
partial credit vs copy-input."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import arc_core as ac
from search import mdl_search
from dsl import apply_program

def train_cellscore(names, train):
    ss = []
    for p in train:
        pr = apply_program(names, p["input"]); gt = p["output"]
        ss.append((pr == gt).mean() if pr.shape == gt.shape else 0.0)
    return float(np.mean(ss))

def main(split="evaluation", tb=1.5):
    tasks = ac.load_tasks(split)
    rows = []; exact = 0; n = 0; used_program = 0
    t0 = time.time()
    for tid, t in tasks.items():
        ranked = mdl_search(t, max_depth=3, beam=80, time_budget=tb)
        best = ranked[0][0] if ranked else []
        cs_best = train_cellscore(best, t["train"])
        cs_id = train_cellscore([], t["train"])
        choice = best if cs_best > cs_id + 1e-9 else []   # conservative: program must STRICTLY beat copy on train
        used_program += int(choice == best and len(best) > 0)
        for tp in t["test"]:
            gt = tp["output"]
            if gt is None: continue
            n += 1
            pr = apply_program(choice, tp["input"])
            s = (pr == gt).mean() if pr.shape == gt.shape else 0.0
            copy = (tp["input"] == gt).mean() if tp["input"].shape == gt.shape else 0.0
            exact += int(s == 1.0)
            rows.append((s, copy))
    A = np.array(rows); dt = time.time() - t0
    nonex = A[A[:, 0] < 1.0]
    print(f"=== C2 graceful (conservative) [{split}]: {n} outputs, {dt:.0f}s ===")
    print(f"used a non-trivial program on {used_program}/{len(tasks)} tasks (else copied input)")
    print(f"EXACT cell-score==1: {exact}/{n} ({100*exact/n:.1f}%)")
    print(f"graceful partial mean cell-score {A[:,0].mean():.3f}  vs copy-input {A[:,1].mean():.3f}  (lift {A[:,0].mean()-A[:,1].mean():+.3f})")
    print(f"on FAILED ({len(nonex)}): graceful {nonex[:,0].mean():.3f} vs copy {nonex[:,1].mean():.3f} (lift {nonex[:,0].mean()-nonex[:,1].mean():+.3f})")
    print(f"failed where graceful>=copy: {int((nonex[:,0]>=nonex[:,1]).sum())}/{len(nonex)} ({100*(nonex[:,0]>=nonex[:,1]).mean():.1f}%)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation")
