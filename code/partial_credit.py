"""C2 — graceful degradation: are the solver's FAILED outputs informative (partial credit)?

Catastrophic failure = a wrong prediction carries 0 information. Graceful = the best-partial
output is measurably closer to the target than trivial guesses. We score attempt_1 (the lowest-MDL
program's output) with shape-aware cell accuracy and compare to trivial baselines (copy-input,
fill-background). Reuses the saved MDL submission (no re-run)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import arc_core as ac

def cell_score(pred, gt):
    """shape-aware: 0 if shape differs, else fraction of cells correct."""
    if pred.shape != gt.shape:
        return 0.0
    return float((pred == gt).mean())

def main(split="evaluation", sub_path="experiments/sub_mdl_evaluation.json"):
    tasks = ac.load_tasks(split)
    with open(sub_path) as f:
        sub = json.load(f)
    rows = []  # (solver, copy_input, fill_bg, shape_match)
    exact = 0; n = 0
    for tid, t in tasks.items():
        pr = sub.get(tid, [])
        for i, tp in enumerate(t["test"]):
            gt = tp["output"]
            if gt is None: continue
            n += 1
            inp = tp["input"]
            a1 = ac.to_grid(pr[i]["attempt_1"]) if i < len(pr) else inp
            a2 = ac.to_grid(pr[i]["attempt_2"]) if i < len(pr) else inp
            s_solver = max(cell_score(a1, gt), cell_score(a2, gt))   # best of the 2 attempts
            s_copy = cell_score(inp, gt)
            bg = int(np.bincount(gt.ravel(), minlength=10).argmax())
            s_bg = cell_score(np.full(gt.shape, bg, dtype=gt.dtype), gt)  # oracle-shape bg fill (upper bound for trivial)
            shape_match = (a1.shape == gt.shape) or (a2.shape == gt.shape)
            exact += int(s_solver == 1.0)
            rows.append((s_solver, s_copy, s_bg, int(shape_match)))
    A = np.array(rows)
    print(f"=== C2 partial credit [{split}]: {n} test outputs ===")
    print(f"EXACT (pass, cell_score==1): {exact}/{n} ({100*exact/n:.1f}%)")
    print(f"solver attempt best  : mean cell-score {A[:,0].mean():.3f} | shape predicted correctly {100*A[:,3].mean():.1f}%")
    print(f"  among shape-correct : mean cell-score {A[A[:,3]==1,0].mean():.3f}  (n={int(A[:,3].sum())})")
    print(f"baseline copy-input  : mean cell-score {A[:,1].mean():.3f}")
    print(f"baseline bg-fill(oracle shape) : mean cell-score {A[:,2].mean():.3f}")
    # partial-credit lift: solver vs best trivial baseline, on tasks we do NOT solve exactly
    nonexact = A[A[:,0] < 1.0]
    base = np.maximum(nonexact[:,1], nonexact[:,2])
    print(f"\nON FAILED tasks ({len(nonexact)}): solver partial {nonexact[:,0].mean():.3f} vs best-trivial {base.mean():.3f} "
          f"(lift {nonexact[:,0].mean()-base.mean():+.3f})")
    print(f"failed tasks where solver partial > best-trivial: {int((nonexact[:,0]>base).sum())}/{len(nonexact)} "
          f"({100*(nonexact[:,0]>base).mean():.1f}%)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evaluation")
