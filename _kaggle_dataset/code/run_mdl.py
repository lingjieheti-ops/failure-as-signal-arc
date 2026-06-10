"""Run the MDL compositional search on a split; report pass@2, exact-program rate, timing."""
import sys, os, time, collections
sys.path.insert(0, os.path.dirname(__file__))
import arc_core as ac
from search import solve_task_mdl

def main(split="evaluation", max_depth=3, beam=80, tb=2.0, limit=None):
    tasks = ac.load_tasks(split)
    items = sorted(tasks.items())
    if limit:
        items = items[:limit]
        tasks = dict(items)
    sub = {}
    n_exact_tasks = 0
    prog_len = collections.Counter()
    t0 = time.time()
    for tid, t in items:
        preds, prog, res, n_exact = solve_task_mdl(t, max_depth=max_depth, beam=beam, time_budget=tb)
        sub[tid] = preds
        if res == 0.0:
            n_exact_tasks += 1
            prog_len[len(prog)] += 1
    dt = time.time() - t0
    m = ac.evaluate(sub, tasks)
    out = f"experiments/sub_mdl_{split}.json"
    ac.save_submission(sub, out)
    print(f"=== MDL search [{split}] depth<={max_depth} beam={beam} tb={tb}s ===")
    print(f"output_pass@2 = {m['output_pass@2']*100:.2f}%  ({m['n_solved_outputs']}/{m['n_outputs']})")
    print(f"task_pass@2   = {m['task_pass@2']*100:.2f}%  ({m['n_solved_tasks']}/{m['n_tasks']})")
    print(f"tasks with EXACT train program: {n_exact_tasks}/{len(items)}  (program-len dist: {dict(prog_len)})")
    print(f"time = {dt:.1f}s ({dt/len(items):.2f}s/task)  -> {out}")

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(split, limit=limit)
