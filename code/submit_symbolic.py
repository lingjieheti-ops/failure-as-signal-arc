"""The official knowledge-free SYMBOLIC entry (no neural, no GPU): MDL compositional search + conservative
anytime policy -> submission.json. attempt_1 = conservative graceful choice (program only if it beats copy on
train); attempt_2 = the top program's output (the gamble), for pass@2 diversity. Reports eval pass@2 and
validates format. This is the documented, reproducible, offline leaderboard entry."""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import arc_core as ac
from search import mdl_search
from dsl import apply_program

def sanitize(g):
    """Guarantee a Kaggle-valid grid: 2D, 1..30 per dim, values 0..9, non-empty."""
    g = np.asarray(g)
    if g.ndim != 2 or g.size == 0:
        return np.zeros((1, 1), dtype=np.int8)
    g = np.clip(g[:30, :30], 0, 9).astype(np.int8)
    if g.shape[0] == 0 or g.shape[1] == 0:
        return np.zeros((1, 1), dtype=np.int8)
    return g

def train_cellscore(names, train):
    ss = []
    for p in train:
        pr = apply_program(names, p["input"]); gt = p["output"]
        ss.append((pr == gt).mean() if pr.shape == gt.shape else 0.0)
    return float(np.mean(ss)) if ss else 0.0

def _factory_rule(task):
    """Parameterized rule families (verified exact on ALL train pairs => zero residual). Training-derived
    (iterations 3-4): counting/ranking/selection + object-property maps. Returns apply fn or None."""
    for mod, name in (("counting", "rule_object_count"), ("objects", "rule_object_map")):
        try:
            fn = getattr(__import__(mod), name)(task)
            if fn is not None:
                return fn
        except Exception:
            continue
    return None

def solve(task, tb=2.0):
    ranked = mdl_search(task, max_depth=3, beam=80, time_budget=tb)
    best = ranked[0][0] if ranked else []
    choice = best if train_cellscore(best, task["train"]) > train_cellscore([], task["train"]) + 1e-9 else []
    fac = _factory_rule(task)  # exact train-fit (residual 0) -> takes precedence as attempt_1
    preds = []
    for tp in task["test"]:
        g = tp["input"]
        if fac is not None:
            try:
                a1 = fac(g)
            except Exception:
                a1 = apply_program(choice, g)
        else:
            a1 = apply_program(choice, g)
        a2 = apply_program(best, g)
        if np.array_equal(a1, a2):  # diversify attempt_2 with next distinct program
            for names, _, _ in ranked[1:]:
                c = apply_program(names, g)
                if c.shape != a1.shape or not np.array_equal(c, a1):
                    a2 = c; break
        preds.append({"attempt_1": ac.to_ll(sanitize(a1)), "attempt_2": ac.to_ll(sanitize(a2))})
    return preds

def main(split="evaluation", out="experiments/submission_symbolic.json"):
    tasks = ac.load_tasks(split)
    sub = {}; t0 = time.time()
    for tid, t in sorted(tasks.items()):
        sub[tid] = solve(t)
    dt = time.time() - t0
    ac.save_submission(sub, out)
    m = ac.evaluate(sub, tasks)
    print(f"=== SYMBOLIC entry [{split}] {m['n_tasks']} tasks, {dt:.0f}s ({dt/m['n_tasks']:.2f}s/task) ===")
    print(f"output_pass@2 {m['output_pass@2']*100:.2f}% ({m['n_solved_outputs']}/{m['n_outputs']})  "
          f"task_pass@2 {m['task_pass@2']*100:.2f}% ({m['n_solved_tasks']}/{m['n_tasks']})")
    print(f"submission -> {out}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="evaluation")
    a = ap.parse_args()
    out = main(a.split)
    import validate_submission
    validate_submission.main(a.split, out)
