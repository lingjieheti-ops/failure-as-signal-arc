"""Kaggle no-internet entry point (the SUBMITTED entry = purely symbolic, knowledge-free, CPU).
Read ARC-AGI-2 test challenges -> MDL search + conservative graceful policy -> format-valid submission.json.
No neural network, no GPU, no internet. Auto-detects the test-challenges file under /kaggle/input.
Finishes in minutes for 120 tasks, far inside the 12h budget."""
import sys, os, json, glob, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arc_core as ac
from submit_symbolic import solve, sanitize  # the official symbolic solver

def load_challenges(path):
    with open(path) as f:
        raw = json.load(f)
    tasks = {}
    for tid, t in raw.items():
        tasks[tid] = {
            "train": [{"input": ac.to_grid(p["input"]), "output": ac.to_grid(p["output"])} for p in t["train"]],
            "test": [{"input": ac.to_grid(p["input"]), "output": None} for p in t["test"]],
        }
    return tasks

def find_challenges():
    for pat in ["/kaggle/input/**/*test*challenge*.json", "/kaggle/input/**/*challenges*test*.json",
                "/kaggle/input/**/*test*.json"]:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    chal = a.challenges or find_challenges()
    out = a.out or ("/kaggle/working/submission.json" if os.path.isdir("/kaggle/working") else "submission.json")
    assert chal and os.path.exists(chal), f"no challenges file (got {chal})"
    print(f"challenges={chal} -> {out}", flush=True)
    tasks = load_challenges(chal)
    sub = {}; t0 = time.time()
    for i, (tid, t) in enumerate(sorted(tasks.items())):
        sub[tid] = solve(t)            # solve() already sanitizes outputs
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(tasks)} ({(time.time()-t0)/(i+1):.2f}s/task)", flush=True)
    with open(out, "w") as f:
        json.dump(sub, f)
    print(f"wrote {out} ({len(sub)} tasks, {time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
