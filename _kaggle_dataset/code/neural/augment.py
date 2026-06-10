"""Knowledge-free task augmentations: D4 (x8) x color-permutation x pair-shuffle.
Semantics-preserving: a correct solver is invariant to all of these. Used for (a) neural training-data
multiplication (~1000x), (b) test-time-augmentation voting at inference. No ARC-specific knowledge."""
import numpy as np

D4_FNS = [
    lambda g: g,
    lambda g: np.rot90(g, 1),
    lambda g: np.rot90(g, 2),
    lambda g: np.rot90(g, 3),
    lambda g: np.fliplr(g),
    lambda g: np.flipud(g),
    lambda g: g.T.copy(),
    lambda g: np.rot90(np.fliplr(g), 1),
]
# inverse D4 index so that D4_FNS[D4_INV[i]](D4_FNS[i](g)) == g
D4_INV = [0, 3, 2, 1, 4, 5, 6, 7]

def color_perm(rng, fix_zero=True):
    p = np.arange(10)
    if fix_zero:
        sub = np.arange(1, 10); rng.shuffle(sub); p[1:] = sub
    else:
        rng.shuffle(p)
    return p

def inv_perm(perm):
    inv = np.zeros_like(perm); inv[perm] = np.arange(len(perm)); return inv

def _ap(g, perm): return perm[g]

def augment_task(task, rng, fix_zero=True):
    """Return (augmented_task, (d4i, perm)) — apply same D4+color map to ALL pairs (train+test)."""
    d4i = int(rng.integers(8)); perm = color_perm(rng, fix_zero); f = D4_FNS[d4i]
    def tp(p):
        out = {"input": _ap(f(p["input"]), perm)}
        out["output"] = _ap(f(p["output"]), perm) if p.get("output") is not None else None
        return out
    return {"train": [tp(p) for p in task["train"]], "test": [tp(p) for p in task["test"]]}, (d4i, perm)

def invert_grid(g, d4i, perm):
    """Map a prediction made in the augmented frame back to the original frame."""
    g = _ap(g, inv_perm(perm))
    return D4_FNS[D4_INV[d4i]](g)

def loo_examples(task):
    """Leave-one-out: yield (demos, query_input, query_output) using each train pair as the query once.
    Multiplies supervision and matches the few-shot inference setup."""
    tr = task["train"]
    for i in range(len(tr)):
        demos = [tr[j] for j in range(len(tr)) if j != i]
        if demos:
            yield demos, tr[i]["input"], tr[i]["output"]
