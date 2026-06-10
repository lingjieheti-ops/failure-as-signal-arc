"""SECOND generality domain — elementary cellular automata (ECA): an EXTERNALLY-DEFINED rule space
(Wolfram's 256 rules) that we did NOT author to match our library. Crucially it contains a natural
NON-REALIZABLE regime (chaotic rules are outside any small local-op library), letting us measure what
Theorem 1 does not cover: behaviour when no primitive closes the gap.

Tasks: hidden rule r in 0..255, one step on width-W periodic binary states; 4 train pairs, 2 held-out tests.
Library: generic local boolean ops (shifts, NOT, pairwise AND/OR/XOR of {left,center,right}, majority).
Measures:
  (1) realizable coverage — how many of the 256 external rules the MDL search solves exactly (test-verified);
  (2) graceful degradation out-of-library — best-program cell accuracy vs copy-input on UNSOLVED rules;
  (3) residual -> closability: does small residual predict that ONE extra primitive (from a held-out pool of
      3-input boolean functions) closes the rule? (the theorem's 'missing single primitive' regime, detected
      from the residual alone).
"""
import sys, os, math, itertools
import numpy as np

W, N_TRAIN, N_TEST, SEED = 15, 4, 2, 0

def ca_step(x, rule):
    l, c, r = np.roll(x, 1), x, np.roll(x, -1)
    idx = (l << 2) | (c << 1) | r
    table = np.array([(rule >> k) & 1 for k in range(8)], dtype=np.uint8)
    return table[idx]

# ---- generic local-op library (NOT authored from the rule table) ----
def L(x): return np.roll(x, 1)
def R(x): return np.roll(x, -1)
BASE = {
    "id": lambda x: x, "shift_l": lambda x: R(x), "shift_r": lambda x: L(x),
    "not": lambda x: 1 - x,
    "and_lr": lambda x: L(x) & R(x), "or_lr": lambda x: L(x) | R(x), "xor_lr": lambda x: L(x) ^ R(x),
    "and_cl": lambda x: x & L(x), "or_cl": lambda x: x | L(x), "xor_cl": lambda x: x ^ L(x),
    "and_cr": lambda x: x & R(x), "or_cr": lambda x: x | R(x), "xor_cr": lambda x: x ^ R(x),
    "maj": lambda x: ((L(x) + x + R(x)) >= 2).astype(np.uint8),
}
# held-out pool: ALL 3-input boolean functions on (l,c,r)... too many (256); use the 70 balanced? Keep it
# principled & cheap: the 16 functions XOR-affine + the 8 single-minterm ANDs + minority = a generic pool
def _minterm(k):
    def f(x, k=k):
        l, c, r = np.roll(x, 1), x, np.roll(x, -1)
        idx = (l << 2) | (c << 1) | r
        return (idx == k).astype(np.uint8)
    return f
POOL = {f"minterm{k}": _minterm(k) for k in range(8)}
POOL["xor3"] = lambda x: L(x) ^ x ^ R(x)
POOL["minority"] = lambda x: ((L(x) + x + R(x)) <= 1).astype(np.uint8)
POOL["nand_lr"] = lambda x: 1 - (L(x) & R(x))
POOL["xnor_lr"] = lambda x: 1 - (L(x) ^ R(x))

def gen_task(rule, rng):
    xs = [rng.integers(0, 2, W).astype(np.uint8) for _ in range(N_TRAIN + N_TEST)]
    pairs = [(x, ca_step(x, rule)) for x in xs]
    return pairs[:N_TRAIN], pairs[N_TRAIN:]

def mdl_search(train, lib, depth=3):
    """BFS over compositions with behavioural dedup; returns (best_prog, residual_cells, best_outputs)."""
    names = list(lib)
    xs = [x for x, _ in train]; ys = [y for _, y in train]
    def res_of(outs):
        return sum(int((o != y).sum()) for o, y in zip(outs, ys))
    frontier = [([], xs)]
    seen = {tuple(np.concatenate(xs).tolist()): 0}
    best = ([], res_of(xs))
    for d in range(depth):
        nxt = []
        for prog, outs in frontier:
            for n in names:
                new = [lib[n](o) for o in outs]
                key = tuple(np.concatenate(new).tolist())
                if key in seen:
                    continue
                seen[key] = 1
                r = res_of(new)
                if (r, len(prog) + 1) < (best[1], len(best[0])):
                    best = (prog + [n], r)
                nxt.append((prog + [n], new))
        frontier = nxt
    return best

def run_prog(prog, x, lib):
    for n in prog:
        x = lib[n](x)
    return x

def main():
    rng = np.random.default_rng(SEED)
    solved, unsolved = [], []
    results = {}
    for rule in range(256):
        train, test = gen_task(rule, rng)
        prog, res = mdl_search(train, BASE, depth=3)
        test_ok = all(np.array_equal(run_prog(prog, x, BASE), y) for x, y in test)
        # graceful partial credit on test (best train program, executed on held-out states)
        pc = float(np.mean([1 - (run_prog(prog, x, BASE) != y).mean() for x, y in test]))
        copy = float(np.mean([1 - (x != y).mean() for x, y in test]))
        results[rule] = dict(res=res, exact=(res == 0 and test_ok), pc=pc, copy=copy, prog=prog)
        (solved if results[rule]["exact"] else unsolved).append(rule)
    print(f"=== ECA domain: 256 external rules, |L|={len(BASE)} generic local ops, depth<=3 ===")
    print(f"(1) REALIZABLE COVERAGE: exact-solved {len(solved)}/256 ({100*len(solved)/256:.0f}%)")
    print(f"    e.g. solved rules: {sorted(solved)[:18]}{' ...' if len(solved)>18 else ''}")
    pcs = np.array([results[r]["pc"] for r in unsolved])
    cps = np.array([results[r]["copy"] for r in unsolved])
    print(f"(2) GRACEFUL on {len(unsolved)} UNSOLVED rules (test states): best-program cell-acc "
          f"{pcs.mean():.3f} vs copy-input {cps.mean():.3f} (lift {pcs.mean()-cps.mean():+.3f}); "
          f"beats-or-ties copy on {(pcs>=cps).mean()*100:.0f}%")
    # (3) residual -> one-more-primitive closability
    tot_cells = N_TRAIN * W
    rows = []
    for rule in unsolved:
        train, test = gen_task(rule, np.random.default_rng(SEED))  # same tasks
        closable = False
        for pname, pfn in POOL.items():
            lib2 = dict(BASE); lib2[pname] = pfn
            prog2, res2 = mdl_search(train, lib2, depth=3)
            if res2 == 0 and all(np.array_equal(run_prog(prog2, x, lib2), y) for x, y in test):
                closable = True
                break
        rows.append((results[rule]["res"] / tot_cells, closable))
    rows = np.array(rows, dtype=float)
    # discrimination: does small residual predict closability?
    cl, ncl = rows[rows[:, 1] == 1, 0], rows[rows[:, 1] == 0, 0]
    print(f"(3) RESIDUAL -> CLOSABILITY on unsolved rules: {int(rows[:,1].sum())}/{len(rows)} closable "
          f"by ONE held-out primitive")
    if len(cl) and len(ncl):
        print(f"    residual fraction: closable mean {cl.mean():.3f} vs not-closable {ncl.mean():.3f}")
        # simple AUC by rank comparison
        auc = float(np.mean([(c < n) + 0.5 * (c == n) for c in cl for n in ncl]))
        print(f"    AUC(residual predicts closability) = {auc:.3f}")
    if len(cl):
        thr = np.median(np.concatenate([cl, ncl])) if len(ncl) else cl.max()
        pred = rows[:, 0] <= thr
        acc = float((pred == rows[:, 1]).mean())
        print(f"    threshold@median: accuracy {acc:.2f}")

if __name__ == "__main__":
    main()
