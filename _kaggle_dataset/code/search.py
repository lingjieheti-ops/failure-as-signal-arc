"""Anytime MDL compositional search (C1) over the generic primitive library.

Objective (description length, lower=better):
    total_bits(program) = program_bits  +  sum_pairs residual_bits(pred, output)
  - program_bits   : Occam prior — longer programs cost more (len * log2|lib|).
  - residual_bits  : bits to correct the train mismatch. Shape mismatch = encode target
                     from scratch (heavy). Shape match = per-wrong-cell (position+value).
                     => a program that gets MOST cells right scores well even if not exact
                        (this is what makes graceful partial output natural — C2).

Beam search by depth, deduplicated by the canonical signature of train predictions
(programs that act identically on train are merged). Anytime: tracks best-so-far under a
per-task time budget. Returns ranked candidates + the best program's train residual (for C3).
"""
from __future__ import annotations
import time, math
from typing import List, Tuple
import numpy as np
from dsl import PRIMS, apply_program

LOG_NC = math.log2(10)
LOG_LIB = math.log2(len(PRIMS))
PRIM_NAMES = [n for n in PRIMS if n != "identity"]  # identity = empty program

def residual_bits(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        oh, ow = gt.shape
        return oh * ow * LOG_NC + 2 * math.log2(31)
    mism = int(np.count_nonzero(pred != gt))
    if mism == 0:
        return 0.0
    H, W = gt.shape
    return mism * (LOG_NC + math.log2(H * W))

def program_bits(names: List[str]) -> float:
    return len(names) * LOG_LIB

def _sig(preds: List[np.ndarray]) -> bytes:
    return b"|".join(bytes(p.shape) + p.tobytes() for p in preds)

def mdl_search(task, max_depth=3, beam=80, time_budget=2.0):
    ins = [p["input"] for p in task["train"]]
    outs = [p["output"] for p in task["train"]]

    def score(names):
        preds = [apply_program(names, x) for x in ins]
        rb = sum(residual_bits(pr, ot) for pr, ot in zip(preds, outs))
        return program_bits(names) + rb, rb, preds

    t0 = time.time()
    evaluated: List[Tuple[List[str], float, float]] = []  # (names, total, residual)
    seen = set()
    frontier = [[]]
    for depth in range(max_depth + 1):
        scored = []
        for names in frontier:
            if time.time() - t0 > time_budget:
                break
            tot, rb, preds = score(names)
            s = _sig(preds)
            if s in seen:
                continue
            seen.add(s)
            evaluated.append((names, tot, rb))
            scored.append((tot, names))
        scored.sort(key=lambda x: x[0])
        nxt = []
        for tot, names in scored[:beam]:
            if len(names) >= max_depth:
                continue
            for p in PRIM_NAMES:
                nxt.append(names + [p])
        frontier = nxt
        if not frontier or time.time() - t0 > time_budget:
            break
    evaluated.sort(key=lambda x: x[1])
    return evaluated  # ranked by total_bits

def solve_task_mdl(task, **kw):
    """Return (preds per test input, best_program, best_residual_bits, n_exact)."""
    ranked = mdl_search(task, **kw)
    n_exact = sum(1 for _, _, rb in ranked if rb == 0.0)
    if not ranked:
        ranked = [([], 0.0, 1e9)]
    best_names = ranked[0][0]
    # attempt_2 = next program whose TEST prediction differs from attempt_1
    preds = []
    for tp in task["test"]:
        g = tp["input"]
        a1 = apply_program(best_names, g)
        a2 = a1
        for names, _, _ in ranked[1:]:
            cand = apply_program(names, g)
            if cand.shape != a1.shape or not np.array_equal(cand, a1):
                a2 = cand
                break
        preds.append({"attempt_1": [[int(v) for v in r] for r in a1],
                      "attempt_2": [[int(v) for v in r] for r in a2]})
    return preds, best_names, ranked[0][2], n_exact
