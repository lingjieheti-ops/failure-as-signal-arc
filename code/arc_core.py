"""ARC-AGI-2 core: IO, grid ops, submission format, faithful local pass@2 evaluator.

Grid = numpy int8 2D array, values 0-9. task_id = filename stem.
Submission schema (Kaggle): {task_id: [ {"attempt_1": grid_ll, "attempt_2": grid_ll}, ... one per test input ]}.
Scoring: per (task, test-input) it is solved if attempt_1 OR attempt_2 exactly equals the
reference output. Kaggle headline score = mean over test-outputs. We also report task-level
(all test outputs of a task correct) since the wording says "percentage of tasks".
"""
from __future__ import annotations
import json, glob, os
from typing import Dict, List, Any
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "ARC-AGI-2", "data")

Grid = np.ndarray  # int8 [H,W]

# ---------- IO ----------
def to_grid(ll: List[List[int]]) -> Grid:
    return np.array(ll, dtype=np.int8)

def to_ll(g: Grid) -> List[List[int]]:
    return [[int(v) for v in row] for row in g]

def load_tasks(split: str) -> Dict[str, dict]:
    """split in {'training','evaluation'}. Returns {tid: {'train':[...], 'test':[...]}} with grids as np arrays."""
    out = {}
    for fp in glob.glob(os.path.join(DATA_ROOT, split, "*.json")):
        tid = os.path.splitext(os.path.basename(fp))[0]
        with open(fp) as f:
            t = json.load(f)
        out[tid] = {
            "train": [{"input": to_grid(p["input"]), "output": to_grid(p["output"])} for p in t["train"]],
            "test":  [{"input": to_grid(p["input"]),
                       "output": to_grid(p["output"]) if "output" in p else None} for p in t["test"]],
        }
    return out

def save_submission(preds: Dict[str, List[dict]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(preds, f)

# ---------- grid ops ----------
def eq(a: Grid, b: Grid) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))

def background(g: Grid) -> int:
    """most frequent color = background heuristic."""
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(cnts))])

# ---------- evaluation ----------
def evaluate(preds: Dict[str, List[dict]], tasks: Dict[str, dict]) -> Dict[str, float]:
    """Returns dict with output-level and task-level pass@2, plus coverage diagnostics."""
    out_total = out_correct = 0
    task_total = task_correct = 0
    missing = 0
    for tid, t in tasks.items():
        gts = [p["output"] for p in t["test"]]
        pr = preds.get(tid)
        task_total += 1
        if pr is None or any(o is None for o in gts):
            missing += 1
            out_total += len(gts)
            continue
        all_ok = True
        for i, gt in enumerate(gts):
            out_total += 1
            ok = False
            if i < len(pr):
                for k in ("attempt_1", "attempt_2"):
                    a = pr[i].get(k)
                    if a is not None and eq(to_grid(a), gt):
                        ok = True
                        break
            out_correct += int(ok)
            all_ok &= ok
        task_correct += int(all_ok)
    return {
        "output_pass@2": out_correct / max(out_total, 1),
        "task_pass@2": task_correct / max(task_total, 1),
        "n_tasks": task_total,
        "n_outputs": out_total,
        "n_solved_outputs": out_correct,
        "n_solved_tasks": task_correct,
        "missing_tasks": missing,
    }

def blank_submission(tasks: Dict[str, dict]) -> Dict[str, List[dict]]:
    """Valid all-zeros submission (every task/test-input has both attempts) — format sanity check."""
    sub = {}
    for tid, t in tasks.items():
        sub[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in t["test"]]
    return sub
