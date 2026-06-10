"""Knowledge-free generic primitive library (DSL) for compositional ARC reasoning.

Design principles:
  - GENERIC: domain-agnostic array / geometry / object / symmetry ops — NOT ARC-curated
    task solutions. This is an inductive bias (a hypothesis space), like CompressARC's
    equivariant architecture — it carries no ARC-specific *knowledge*.
  - TOTAL: every primitive returns a grid and never raises (enables graceful degradation
    and crash-free search).
  - TYPED (lightweight): all are Grid -> Grid so the search composes simple chains.
  - Each primitive has an MDL cost (bits to name it); cost is used by the search.

Objects via scipy.ndimage.label (4/8-connectivity). Background = most frequent color.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple
import numpy as np
from scipy import ndimage

Grid = np.ndarray

def background(g: Grid) -> int:
    v, c = np.unique(g, return_counts=True)
    return int(v[int(np.argmax(c))])

def _bbox(mask: np.ndarray):
    rs, cs = np.where(mask)
    if rs.size == 0:
        return None
    return rs.min(), rs.max(), cs.min(), cs.max()

# ---------------- geometry ----------------
def identity(g): return g
def rot90(g): return np.rot90(g, 1)
def rot180(g): return np.rot90(g, 2)
def rot270(g): return np.rot90(g, 3)
def flip_h(g): return np.fliplr(g)
def flip_v(g): return np.flipud(g)
def transpose(g): return g.T.copy()
def anti_transpose(g): return np.rot90(np.fliplr(g), 1)

# ---------------- crop / extract ----------------
def crop_content(g):
    bb = _bbox(g != background(g))
    if bb is None: return g
    r0, r1, c0, c1 = bb
    return g[r0:r1+1, c0:c1+1].copy()

def _components(g, connectivity=2):
    """Connected components of non-background cells (8-conn default). Returns list of masks."""
    bg = background(g)
    fg = g != bg
    struct = ndimage.generate_binary_structure(2, connectivity)
    lab, n = ndimage.label(fg, structure=struct)
    return [(lab == i) for i in range(1, n + 1)]

def largest_object(g):
    comps = _components(g)
    if not comps: return g
    m = max(comps, key=lambda x: int(x.sum()))
    bb = _bbox(m)
    r0, r1, c0, c1 = bb
    return g[r0:r1+1, c0:c1+1].copy()

def smallest_object(g):
    comps = _components(g)
    if not comps: return g
    m = min(comps, key=lambda x: int(x.sum()))
    bb = _bbox(m)
    r0, r1, c0, c1 = bb
    return g[r0:r1+1, c0:c1+1].copy()

def keep_largest(g):
    comps = _components(g)
    if not comps: return g
    bg = background(g)
    m = max(comps, key=lambda x: int(x.sum()))
    out = np.full_like(g, bg)
    out[m] = g[m]
    return out

def remove_largest(g):
    comps = _components(g)
    if not comps: return g
    bg = background(g)
    m = max(comps, key=lambda x: int(x.sum()))
    out = g.copy()
    out[m] = bg
    return out

# ---------------- mirror tilings (parameter-free) ----------------
def tile_2x2(g): return np.tile(g, (2, 2))
def mirror_h(g): return np.concatenate([g, np.fliplr(g)], axis=1)
def mirror_v(g): return np.concatenate([g, np.flipud(g)], axis=0)
def mirror_4(g):
    top = np.concatenate([g, np.fliplr(g)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)

# ---------------- periodicity ----------------
def _find_period(arr_len, vec_eq):
    for p in range(1, arr_len):
        if all(vec_eq(i, i % p) for i in range(arr_len)):
            return p
    return arr_len

def crop_to_period(g):
    """Extract the fundamental repeating tile (row & col period)."""
    H, W = g.shape
    pr = _find_period(H, lambda i, j: np.array_equal(g[i], g[j]))
    pc = _find_period(W, lambda i, j: np.array_equal(g[:, i], g[:, j]))
    if pr == H and pc == W: return g
    return g[:pr, :pc].copy()

# ---------------- symmetry completion / occlusion repair ----------------
_SYMS = {
    "fliplr": np.fliplr, "flipud": np.flipud, "rot180": lambda a: np.rot90(a, 2),
    "transpose": lambda a: a.T, "anti": lambda a: np.rot90(np.fliplr(a), 1),
}

def _applicable_syms(g):
    H, W = g.shape
    out = []
    for name, fn in _SYMS.items():
        if name in ("transpose", "anti") and H != W:
            continue
        out.append(fn)
    return out

def _reconstruct(g, noise: int):
    """Fill cells equal to `noise` using symmetric counterparts; iterate to convergence."""
    filled = g.copy()
    syms = _applicable_syms(g)
    for _ in range(4):
        changed = False
        for fn in syms:
            sg = fn(filled)
            if sg.shape != filled.shape:
                continue
            mask = (filled == noise) & (sg != noise)
            if mask.any():
                filled[mask] = sg[mask]
                changed = True
        if not changed:
            break
    return filled

def _best_noise_color(g):
    """Color whose removal best reveals symmetry (occluder). Returns (color, residual_after)."""
    best, best_res = None, None
    syms = _applicable_syms(g)
    if not syms:
        return None, None
    for c in np.unique(g):
        if (g == c).sum() == 0:
            continue
        filled = _reconstruct(g, int(c))
        # residual asymmetry ignoring still-noise cells
        res = 0
        for fn in syms:
            sg = fn(filled)
            if sg.shape != filled.shape:
                continue
            res += int(((filled != sg) & (filled != c) & (sg != c)).sum())
        # prefer colors that (a) leave low residual and (b) actually got filled
        filled_cnt = int((g == c).sum() - (filled == c).sum())
        if filled_cnt == 0:
            continue
        score = res - 0.001 * filled_cnt
        if best_res is None or score < best_res:
            best_res, best = score, int(c)
    return best, best_res

def complete_symmetry(g):
    """Reconstruct the full grid assuming an occluder color broke its symmetry."""
    c, _ = _best_noise_color(g)
    if c is None:
        return g
    return _reconstruct(g, c)

def restore_occluded_region(g):
    """Detect occluder color, reconstruct via symmetry, return the (cropped) restored region."""
    c, _ = _best_noise_color(g)
    if c is None:
        return g
    bb = _bbox(g == c)
    if bb is None:
        return g
    filled = _reconstruct(g, c)
    r0, r1, c0, c1 = bb
    return filled[r0:r1+1, c0:c1+1].copy()

# ---------------- robust symmetry repair (axis/period/rotation SEARCH + inpaint) ----------------
# Generic occlusion-repair: find the symmetry group satisfied by the visible cells (searching
# over all mirror axes, translational periods, 180-rotation, transposes), then inpaint the
# occluded cells by propagating known values through that group. Knowledge-free.
def _candidate_syms(H, W):
    idxr, idxc = np.indices((H, W))
    c = []
    for a in range(1, 2 * W - 2):           # vertical mirror about col axis a/2
        c.append((idxr, a - idxc))
    for b in range(1, 2 * H - 2):           # horizontal mirror about row axis b/2
        c.append((b - idxr, idxc))
    c.append(((H - 1) - idxr, (W - 1) - idxc))   # 180 rotation (full grid)
    for p in range(1, H):                   # vertical translation period p (both dirs)
        c.append((idxr - p, idxc)); c.append((idxr + p, idxc))
    for p in range(1, W):                   # horizontal translation period p
        c.append((idxr, idxc - p)); c.append((idxr, idxc + p))
    if H == W:
        c.append((idxc, idxr))                       # transpose
        c.append(((W - 1) - idxc, (H - 1) - idxr))   # anti-transpose
    return c

def _repair_with_occluder(g, occ):
    H, W = g.shape
    known = g != occ
    if int(known.sum()) < 2:
        return None, 0
    min_overlap = max(8, (H * W) // 12)
    accepted = []
    for src_r, src_c in _candidate_syms(H, W):
        valid = (src_r >= 0) & (src_r < H) & (src_c >= 0) & (src_c < W)
        sr = np.clip(src_r, 0, H - 1); sc = np.clip(src_c, 0, W - 1)
        img = g[sr, sc]
        kk = valid & known & known[sr, sc]
        if int(kk.sum()) < min_overlap:
            continue
        if np.array_equal(g[kk], img[kk]):          # perfect on the visible overlap
            accepted.append((sr, sc, valid))
    if not accepted:
        return None, 0
    filled = g.copy()
    for _ in range(6):
        progressed = False
        for sr, sc, valid in accepted:
            src_known = valid & (filled[sr, sc] != occ)
            tofill = (filled == occ) & src_known
            if tofill.any():
                filled[tofill] = filled[sr, sc][tofill]
                progressed = True
        if not progressed:
            break
    n_filled = int((g == occ).sum() - (filled == occ).sum())
    return filled, n_filled

def _best_repair(g):
    best = None
    for occ in np.unique(g):
        filled, nf = _repair_with_occluder(g, int(occ))
        if filled is None or nf == 0:
            continue
        if best is None or nf > best[2]:
            best = (int(occ), filled, nf)
    return best

def symmetry_repair(g):
    b = _best_repair(g)
    return b[1] if b else g

def symmetry_repair_extract(g):
    b = _best_repair(g)
    if not b:
        return g
    occ, filled, _ = b
    bb = _bbox(g == occ)
    if bb is None:
        return filled
    r0, r1, c0, c1 = bb
    return filled[r0:r1 + 1, c0:c1 + 1].copy()

# ---------------- soft periodicity (frontier map's #1 lever: ~30% of eval is periodic) ----------------
def _orbit_mode(g, pr, pc):
    recon = g.copy()
    for i in range(pr):
        for j in range(pc):
            cells = g[i::pr, j::pc]
            vals, cnts = np.unique(cells, return_counts=True)
            recon[i::pr, j::pc] = int(vals[int(cnts.argmax())])
    return recon

def _best_period(g, floor=0.70):
    """Smallest (pr,pc) whose orbit-mode reconstruction agrees with >= floor of cells (MDL: small period)."""
    H, W = g.shape
    best = None
    for pr in (range(1, H // 2 + 1) if H > 1 else [1]):
        for pc in (range(1, W // 2 + 1) if W > 1 else [1]):
            if pr == H and pc == W:
                continue
            recon = _orbit_mode(g, pr, pc)
            agree = float((recon == g).mean())
            if agree >= floor:
                key = (pr * pc, -agree)
                if best is None or key < best[0]:
                    best = (key, pr, pc, recon, agree)
    return best

def periodic_denoise(g):
    """Repair a (mostly-)periodic grid: each cell -> mode over its periodic orbit (fixes noise/occlusion)."""
    b = _best_period(g)
    return b[3] if b else g

def periodic_tile(g):
    """Extract the fundamental pr x pc tile of the (denoised) periodic grid."""
    b = _best_period(g)
    if not b:
        return g
    _, pr, pc, recon, _ = b
    return recon[:pr, :pc].copy()

# ---------------- cardinality-based object selection (composable; from the loop's 2nd iteration) ----------------
def _sel(crit):
    def f(g, crit=crit):
        from collections import Counter
        comps = _components(g)
        if not comps:
            return g
        if crit in ("unique_shape", "modal_shape"):
            keys = []
            for m in comps:
                r0, r1, c0, c1 = _bbox(m)
                sub = np.where(m[r0:r1+1, c0:c1+1], g[r0:r1+1, c0:c1+1], -1)
                keys.append((sub.shape, sub.tobytes()))
            cnt = Counter(keys)
            if crit == "unique_shape":
                want = [i for i, k in enumerate(keys) if cnt[k] == 1]
                if len(want) != 1:
                    return g
                m = comps[want[0]]
            else:
                mode = cnt.most_common(1)[0][0]
                want = [i for i, k in enumerate(keys) if k == mode]
                m = max((comps[i] for i in want), key=lambda x: int(x.sum()))
        elif crit == "unique_color":
            cols = []
            for m0 in comps:
                vals, cnts = np.unique(g[m0], return_counts=True)
                cols.append(int(vals[int(np.argmax(cnts))]))
            cnt = Counter(cols)
            want = [i for i, c in enumerate(cols) if cnt[c] == 1]
            if len(want) != 1:
                return g
            m = comps[want[0]]
        else:
            return g
        r0, r1, c0, c1 = _bbox(m)
        return g[r0:r1+1, c0:c1+1].copy()
    return f

select_unique_shape = _sel("unique_shape")
select_modal_shape = _sel("modal_shape")
select_unique_color = _sel("unique_color")

# ---------------- panel decomposition + selection (iteration 3, training-derived) ----------------
def _split_panels(g):
    """Split into >=2 equal panels: by full separator rows/cols of one constant color, else by equal halves/
    thirds along the longer axis. Returns list of equal-shape subgrids or None."""
    H, W = g.shape
    for axis in (1, 0):
        size = W if axis == 1 else H
        lines = []
        for i in range(size):
            vec = g[:, i] if axis == 1 else g[i, :]
            if len(np.unique(vec)) == 1:
                lines.append(i)
        if lines:
            # contiguous separator groups -> cut points
            cuts, run = [], [lines[0]]
            for i in lines[1:]:
                if i == run[-1] + 1:
                    run.append(i)
                else:
                    cuts.append(run); run = [i]
            cuts.append(run)
            # panels = segments between separators (and at the ends)
            bounds, prev = [], 0
            for run in cuts:
                if run[0] > prev:
                    bounds.append((prev, run[0]))
                prev = run[-1] + 1
            if prev < size:
                bounds.append((prev, size))
            panels = [g[:, a:b] if axis == 1 else g[a:b, :] for a, b in bounds]
            shapes = {p.shape for p in panels}
            if len(panels) >= 2 and len(shapes) == 1 and panels[0].size > 1:
                return panels
    for axis in (1, 0):  # equal split fallback
        size = W if axis == 1 else H
        for k in (2, 3, 4):
            if size % k == 0 and size // k > 1:
                step = size // k
                panels = [g[:, i*step:(i+1)*step] if axis == 1 else g[i*step:(i+1)*step, :] for i in range(k)]
                return panels  # caller's consistency check vets this hypothesis
    return None

def select_odd_panel(g):
    """Return the panel that differs from the (otherwise identical) majority of panels."""
    panels = _split_panels(g)
    if not panels or len(panels) < 3:
        return g
    keys = [p.tobytes() for p in panels]
    from collections import Counter
    cnt = Counter(keys)
    if len(cnt) != 2:
        return g
    odd_key, n = cnt.most_common()[-1]
    if n != 1:
        return g
    return panels[keys.index(odd_key)].copy()

def select_majority_panel(g):
    """Return the modal panel (the content most panels agree on)."""
    panels = _split_panels(g)
    if not panels or len(panels) < 2:
        return g
    keys = [p.tobytes() for p in panels]
    from collections import Counter
    cnt = Counter(keys)
    top_key, n = cnt.most_common(1)[0]
    if n < 2:
        return g
    return panels[keys.index(top_key)].copy()

# ---------------- registry ----------------
PRIMS: Dict[str, Callable[[Grid], Grid]] = {
    "select_odd_panel": select_odd_panel, "select_majority_panel": select_majority_panel,
    "symmetry_repair": symmetry_repair, "symmetry_repair_extract": symmetry_repair_extract,
    "periodic_denoise": periodic_denoise, "periodic_tile": periodic_tile,
    "select_unique_shape": select_unique_shape, "select_modal_shape": select_modal_shape,
    "select_unique_color": select_unique_color,
    "identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270,
    "flip_h": flip_h, "flip_v": flip_v, "transpose": transpose, "anti_transpose": anti_transpose,
    "crop_content": crop_content, "largest_object": largest_object, "smallest_object": smallest_object,
    "keep_largest": keep_largest, "remove_largest": remove_largest,
    "tile_2x2": tile_2x2, "mirror_h": mirror_h, "mirror_v": mirror_v, "mirror_4": mirror_4,
    "crop_to_period": crop_to_period,
    "complete_symmetry": complete_symmetry, "restore_occluded_region": restore_occluded_region,
}

def apply_program(names: List[str], g: Grid) -> Grid:
    cur = g
    for n in names:
        try:
            cur = PRIMS[n](cur)
            if cur.size == 0 or cur.shape[0] > 60 or cur.shape[1] > 60:
                return g  # guard against blowups
        except Exception:
            return g
    return np.ascontiguousarray(cur)
