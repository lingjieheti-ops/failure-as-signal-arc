"""HYBRID solver (the method): symbolic-exact-first, neural-fallback, diagnostic-routed.

Per task:
  - run the knowledge-free MDL symbolic search. If it finds a program with ZERO train residual
    -> that rule is exact + interpretable -> use it (attempt_1), neural as attempt_2.
  - else -> trust the neural net (attempt_1); attempt_2 = symbolic best-partial IF it beats copy-input
    on train (conservative), else neural's 2nd candidate.
This keeps C1/C2/C3 + frontier map intact and adds the neural accuracy engine. Reports how often each
route fires (interpretability metric)."""
import sys, os, argparse, time, collections
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
import arc_core as ac
from model import GPT
from tokenizer import VOCAB
from search import mdl_search
from dsl import apply_program
import infer

def train_cellscore(names, train):
    ss = []
    for p in train:
        pr = apply_program(names, p["input"]); gt = p["output"]
        ss.append((pr == gt).mean() if pr.shape == gt.shape else 0.0)
    return float(np.mean(ss)) if ss else 0.0

def solve_hybrid(model, task, dev, max_len, n_aug, route):
    ranked = mdl_search(task, max_depth=3, beam=80, time_budget=1.0)
    exact = bool(ranked) and ranked[0][2] == 0.0
    sym = ranked[0][0] if ranked else []
    sym_beats_copy = train_cellscore(sym, task["train"]) > train_cellscore([], task["train"]) + 1e-9
    neural = infer.solve_task(model, task, dev, max_len, n_aug)
    out = []
    for ti, tp in enumerate(task["test"]):
        npred = neural[ti]
        sp = ac.to_ll(apply_program(sym, tp["input"]))
        if exact:
            route["symbolic_exact"] += 1
            a1, a2 = sp, npred["attempt_1"]
        else:
            route["neural"] += 1
            a1 = npred["attempt_1"]
            a2 = sp if sym_beats_copy else npred["attempt_2"]
        out.append({"attempt_1": a1, "attempt_2": a2})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/gpt.pt")
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_aug", type=int, default=3)
    ap.add_argument("--out", default="experiments/sub_hybrid.json")
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev); cfg = ck["cfg"]
    model = GPT(VOCAB, cfg["d"], cfg["heads"], cfg["layers"], cfg["max_len"]).to(dev).eval()
    model.load_state_dict(ck["sd"])
    tasks = ac.load_tasks(a.split); items = sorted(tasks.items())
    if a.limit: items = items[:a.limit]
    route = collections.Counter(); sub = {}; t0 = time.time()
    for i, (tid, t) in enumerate(items):
        sub[tid] = solve_hybrid(model, t, dev, cfg["max_len"], a.n_aug, route)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(items)} ({(time.time()-t0)/(i+1):.1f}s/task)", flush=True)
    ac.save_submission(sub, a.out)
    m = ac.evaluate(sub, {k: tasks[k] for k, _ in items})
    print(f"=== HYBRID [{a.split}] {len(items)} tasks, {time.time()-t0:.0f}s ===")
    print(f"output_pass@2 {m['output_pass@2']*100:.2f}%  task_pass@2 {m['task_pass@2']*100:.2f}%")
    print(f"routes: {dict(route)}  -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
