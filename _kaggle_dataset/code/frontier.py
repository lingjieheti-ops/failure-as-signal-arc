"""Frontier map (novel contribution): cluster ARC-AGI-2 tasks by a structural residual-signature so
the 95% the diagnostic can't NAME are still ORGANIZED into reusable meta-families. Output = a
prioritized map of what abstraction the field still needs (a roadmap toward 85%).

Each task -> a feature vector computed from its train pairs (shape relation, palette change, input/output
symmetry & periodicity, object counts, locality, extraction). Standardize -> k-means (scipy). Report
cluster sizes + the features that most distinguish each cluster + example task ids.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.cluster.vq import kmeans2, whiten
import arc_core as ac
from dsl import crop_to_period, _components, background

def _cols(g): return set(int(v) for v in np.unique(g))
def _hsym(g): return float(np.array_equal(g, np.fliplr(g)))
def _vsym(g): return float(np.array_equal(g, np.flipud(g)))
def _rot(g):  return float(g.shape[0]==g.shape[1] and np.array_equal(g, np.rot90(g,2)))
def _periodic(g):
    try: return float(crop_to_period(g).size < g.size)
    except Exception: return 0.0
def _is_subgrid(s, b):
    sh,sw=s.shape; bh,bw=b.shape
    if sh>bh or sw>bw: return 0.0
    for r in range(bh-sh+1):
        for c in range(bw-sw+1):
            if np.array_equal(b[r:r+sh,c:c+sw], s): return 1.0
    return 0.0

FEATS = ["area_ratio","h_ratio","w_ratio","same_shape","ncol_in","ncol_out","ncol_delta",
         "bg_changed","in_hsym","in_vsym","in_rot","out_hsym","out_vsym","in_periodic",
         "out_periodic","nobj_in","nobj_out","nobj_delta","cellchange","out_subgrid_in"]

def features(task):
    acc = collections.defaultdict(list)
    for p in task["train"]:
        i, o = p["input"], p["output"]
        ih,iw=i.shape; oh,ow=o.shape
        acc["area_ratio"].append((oh*ow)/max(ih*iw,1))
        acc["h_ratio"].append(oh/max(ih,1)); acc["w_ratio"].append(ow/max(iw,1))
        acc["same_shape"].append(float(i.shape==o.shape))
        ci,co=_cols(i),_cols(o)
        acc["ncol_in"].append(len(ci)); acc["ncol_out"].append(len(co))
        acc["ncol_delta"].append(len(co)-len(ci))
        acc["bg_changed"].append(float(background(i)!=background(o)))
        acc["in_hsym"].append(_hsym(i)); acc["in_vsym"].append(_vsym(i)); acc["in_rot"].append(_rot(i))
        acc["out_hsym"].append(_hsym(o)); acc["out_vsym"].append(_vsym(o))
        acc["in_periodic"].append(_periodic(i)); acc["out_periodic"].append(_periodic(o))
        no_i,no_o=len(_components(i)),len(_components(o))
        acc["nobj_in"].append(no_i); acc["nobj_out"].append(no_o); acc["nobj_delta"].append(no_o-no_i)
        acc["cellchange"].append(float((i!=o).mean()) if i.shape==o.shape else 1.0)
        acc["out_subgrid_in"].append(_is_subgrid(o,i) if o.size<=i.size else 0.0)
    return np.array([np.mean(acc[f]) for f in FEATS], dtype=float)

def main(split="evaluation", k=7, seed=0):
    tasks = ac.load_tasks(split)
    tids = sorted(tasks)
    X = np.array([features(tasks[t]) for t in tids])
    Xw = whiten(np.nan_to_num(X))
    cent, lab = kmeans2(Xw, k, seed=seed, minit="++", missing="warn")
    gmean = Xw.mean(0)
    print(f"=== frontier map [{split}]: {len(tids)} tasks, k={k} clusters ===")
    order = sorted(range(k), key=lambda c: -(lab==c).sum())
    for c in order:
        idx = np.where(lab==c)[0]
        if len(idx)==0: continue
        cmean = Xw[idx].mean(0)
        dev = cmean - gmean
        top = sorted(range(len(FEATS)), key=lambda j: -abs(dev[j]))[:4]
        sig = ", ".join(f"{FEATS[j]}{'+' if dev[j]>0 else '-'}{abs(X[idx][:,j].mean()):.2f}" for j in top)
        ex = ", ".join(tids[i] for i in idx[:3])
        print(f"  cluster {c}: n={len(idx):3d} ({100*len(idx)/len(tids):4.1f}%) | {sig} | e.g. {ex}")

def _ari(a, b):
    """Adjusted Rand Index between two labelings (no sklearn dependency)."""
    import collections
    n = len(a)
    pairs = lambda x: x * (x - 1) / 2.0
    ct = collections.Counter(zip(a, b)); A = collections.Counter(a); B = collections.Counter(b)
    s_ij = sum(pairs(c) for c in ct.values())
    s_a = sum(pairs(c) for c in A.values()); s_b = sum(pairs(c) for c in B.values())
    exp = s_a * s_b / pairs(n); mx = (s_a + s_b) / 2.0
    return 1.0 if mx == exp else (s_ij - exp) / (mx - exp)

def stability(split="evaluation", k=7, n_seeds=20):
    """Cluster-stability: mean pairwise ARI across seeds at k; and ARI of k=7 vs k in 5..9 (seed 0)."""
    tasks = ac.load_tasks(split)
    tids = sorted(tasks)
    X = np.array([features(tasks[t]) for t in tids])
    Xw = whiten(np.nan_to_num(X))
    labs = [kmeans2(Xw, k, seed=s, minit="++", missing="warn")[1] for s in range(n_seeds)]
    aris = [_ari(labs[i], labs[j]) for i in range(n_seeds) for j in range(i + 1, n_seeds)]
    print(f"=== stability [{split}] k={k}, {n_seeds} seeds ===")
    print(f"pairwise ARI across seeds: mean {np.mean(aris):.3f} +- {np.std(aris):.3f} (min {np.min(aris):.3f})")
    base = labs[0]
    for kk in (5, 6, 8, 9):
        lk = kmeans2(Xw, kk, seed=0, minit="++", missing="warn")[1]
        print(f"ARI(k={k} vs k={kk}): {_ari(base, lk):.3f}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stability":
        stability(sys.argv[2] if len(sys.argv) > 2 else "evaluation",
                  int(sys.argv[3]) if len(sys.argv) > 3 else 7)
    else:
        main(sys.argv[1] if len(sys.argv)>1 else "evaluation",
             int(sys.argv[2]) if len(sys.argv)>2 else 7)
