"""Run a solver on a split, report pass@2 metrics + timing, save submission.json."""
import sys, os, time, json, collections
sys.path.insert(0, os.path.dirname(__file__))
import arc_core as ac
from solver_baseline import solve_task

def main(split="evaluation", out="experiments/sub_baseline_eval.json"):
    tasks = ac.load_tasks(split)
    sub = {}
    rule_hist = collections.Counter()
    solved_by_rule = collections.Counter()
    t0 = time.time()
    for tid, t in sorted(tasks.items()):
        preds, names = solve_task(t)
        sub[tid] = preds
        for n in names:
            rule_hist[n] += 1
    dt = time.time() - t0

    m = ac.evaluate(sub, tasks)
    # attribute solves to top consistent rule (rough)
    for tid, t in tasks.items():
        preds, names = solve_task(t)
        gts = [p["output"] for p in t["test"]]
        ok = all(any(o is not None and ac.eq(ac.to_grid(preds[i][k]), o)
                     for k in ("attempt_1", "attempt_2")) for i, o in enumerate(gts))
        if ok and names:
            solved_by_rule[names[0]] += 1

    os.makedirs(os.path.dirname(out), exist_ok=True)
    ac.save_submission(sub, out)

    print(f"=== {split}: {m['n_tasks']} tasks, {m['n_outputs']} outputs ===")
    print(f"output_pass@2 = {m['output_pass@2']*100:.2f}%  ({m['n_solved_outputs']}/{m['n_outputs']})")
    print(f"task_pass@2   = {m['task_pass@2']*100:.2f}%  ({m['n_solved_tasks']}/{m['n_tasks']})")
    print(f"time = {dt:.2f}s  ({1000*dt/max(m['n_tasks'],1):.1f} ms/task)")
    print(f"submission -> {out}")
    print("\nrules consistent-with-train (count of tasks):")
    for n, c in rule_hist.most_common():
        print(f"  {n:16s} {c}")
    print("\nsolved tasks attributed to top rule:")
    for n, c in solved_by_rule.most_common():
        print(f"  {n:16s} {c}")

if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    main(split)
