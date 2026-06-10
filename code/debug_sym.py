"""Diagnose the true symmetry structure of 0934a4d8 (an occlusion-repair eval task)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, arc_core as ac

tasks = ac.load_tasks("evaluation")
t = tasks["0934a4d8"]
g = t["train"][0]["input"]; out = t["train"][0]["output"]
print("input shape", g.shape, "output shape", out.shape)
vals, cnts = np.unique(g, return_counts=True)
print("colors/counts:", dict(zip(vals.tolist(), cnts.tolist())))

# candidate occluder = color forming a solid rectangle
for c in vals:
    rs, cs = np.where(g == c)
    if rs.size == 0: continue
    bb_area = (rs.max()-rs.min()+1)*(cs.max()-cs.min()+1)
    solid = rs.size == bb_area
    if solid and rs.size > 1:
        print(f"  color {c}: SOLID rect {rs.min()}-{rs.max()} x {cs.min()}-{cs.max()} ({rs.size} cells)")

noise = 8
mask = g != noise
def sym_score(fn):
    sg = fn(g)
    if sg.shape != g.shape: return None
    both = mask & (sg != noise)
    if both.sum() == 0: return None
    return float((g[both] == sg[both]).mean()), int(both.sum())
print("\nGLOBAL symmetry scores (ignoring noise=8):")
for name, fn in [("fliplr",np.fliplr),("flipud",np.flipud),("rot180",lambda a:np.rot90(a,2)),
                 ("transpose",lambda a:a.T),("anti",lambda a:np.rot90(np.fliplr(a),1))]:
    print(f"  {name}: {sym_score(fn)}")

# translational periodicity ignoring noise
H, W = g.shape
def row_period():
    for p in range(1, H):
        ok = True
        for i in range(H):
            j = i % p
            m = (g[i]!=noise) & (g[j]!=noise)
            if m.any() and not np.array_equal(g[i][m], g[j][m]): ok=False; break
        if ok: return p
    return H
def col_period():
    for p in range(1, W):
        ok = True
        for i in range(W):
            j = i % p
            m = (g[:,i]!=noise)&(g[:,j]!=noise)
            if m.any() and not np.array_equal(g[:,i][m], g[:,j][m]): ok=False; break
        if ok: return p
    return W
print(f"\nrow period (ignoring noise): {row_period()} / {H}")
print(f"col period (ignoring noise): {col_period()} / {W}")

# off-center mirror: try mirror about every possible vertical axis
def best_vmirror():
    best=None
    for axis2 in range(1, 2*W-2):  # axis at axis2/2
        diff=0; cnt=0
        for cc in range(W):
            mc = axis2-cc
            if 0<=mc<W:
                m=(g[:,cc]!=noise)&(g[:,mc]!=noise)
                diff+=int((g[:,cc][m]!=g[:,mc][m]).sum()); cnt+=int(m.sum())
        if cnt>0 and (best is None or diff/cnt<best[1]): best=(axis2,diff/cnt,cnt)
    return best
print("best vertical mirror axis (axis*2, err_rate, cnt):", best_vmirror())
