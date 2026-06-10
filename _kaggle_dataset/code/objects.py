"""Object-map rule families (sprint 4, clean-room: training-derived). The frontier map says ~56% of eval is
same-shape editing; the plausible mechanism is OBJECT-LEVEL transformation. These factories learn, per task,
a mapping from a generic object PROPERTY to an action (recolor / keep / remove), verified on ALL train pairs.

Generic properties (no ARC knowledge): size (cell count), size rank, canonical shape class, border contact,
hole count. Actions: recolor-by-property-map, filter (keep/remove objects whose property maps to 'delete').
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from arc_core import eq
from dsl import background, _bbox

def _comps(g, connectivity=2, same_color=True):
    bg = background(g)
    struct = ndimage.generate_binary_structure(2, connectivity)
    masks = []
    if same_color:
        for c in np.unique(g):
            if int(c) == bg:
                continue
            lab, n = ndimage.label(g == c, structure=struct)
            masks.extend([(lab == i) for i in range(1, n + 1)])
    else:
        lab, n = ndimage.label(g != bg, structure=struct)
        masks = [(lab == i) for i in range(1, n + 1)]
    return masks, bg

def _shape_key(m):
    r0, r1, c0, c1 = _bbox(m)
    sub = m[r0:r1+1, c0:c1+1]
    # canonical under D4
    cands = []
    for k in range(4):
        r = np.rot90(sub, k)
        cands.append(r.tobytes() + bytes(r.shape))
        f = np.fliplr(r)
        cands.append(f.tobytes() + bytes(f.shape))
    return min(cands)

def _holes(m):
    # background components fully inside the bbox not touching its border
    r0, r1, c0, c1 = _bbox(m)
    sub = ~m[r0:r1+1, c0:c1+1]
    lab, n = ndimage.label(sub)
    holes = 0
    H, W = sub.shape
    for i in range(1, n + 1):
        comp = lab == i
        rs, cs = np.where(comp)
        if rs.min() > 0 and rs.max() < H - 1 and cs.min() > 0 and cs.max() < W - 1:
            holes += 1
    return holes

def _props(g, masks):
    H, W = g.shape
    sizes = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: -sizes[i])
    rank = {i: r for r, i in enumerate(order)}
    out = []
    for i, m in enumerate(masks):
        rs, cs = np.where(m)
        vals, cnts = np.unique(g[m], return_counts=True)
        color = int(vals[int(np.argmax(cnts))])
        out.append({
            "size": sizes[i],
            "size_rank": rank[i],
            "shape": _shape_key(m),
            "border": int(rs.min() == 0 or cs.min() == 0 or rs.max() == H - 1 or cs.max() == W - 1),
            "holes": _holes(m),
            "color": color,
        })
    return out

PROPS = ["size", "size_rank", "shape", "border", "holes", "color"]

def rf_object_map(pairs):
    """Learn property -> new_color (or 'DEL') from train pairs; apply objectwise. Same-shape tasks only.
    For each (connectivity, same_color, property): build the map from every train pair; verify exactly."""
    if not all(p["input"].shape == p["output"].shape for p in pairs):
        return []
    for conn in (2, 1):
        for bycolor in (True, False):
            for prop in PROPS:
                mapping = {}
                ok = True
                for p in pairs:
                    g, o = p["input"], p["output"]
                    masks, bg = _comps(g, conn, bycolor)
                    if not masks:
                        ok = False; break
                    props = _props(g, masks)
                    covered = np.zeros_like(g, dtype=bool)
                    for m, pr in zip(masks, props):
                        covered |= m
                        ovals = o[m]
                        key = pr[prop]
                        if np.all(ovals == ovals.ravel()[0]):
                            tgt = int(ovals.ravel()[0])
                            act = "DEL" if tgt == background(g) else tgt
                        else:
                            ok = False; break
                        if key in mapping and mapping[key] != act:
                            ok = False; break
                        mapping[key] = act
                    if not ok:
                        break
                    # background cells must be unchanged
                    if not np.array_equal(g[~covered], o[~covered]):
                        ok = False; break
                if not ok or len(mapping) < 2:
                    continue
                def apply(g, conn=conn, bycolor=bycolor, prop=prop, mp=dict(mapping)):
                    masks, bg = _comps(g, conn, bycolor)
                    out = g.copy()
                    for m, pr in zip(masks, _props(g, masks)):
                        key = pr[prop]
                        if key not in mp:
                            raise ValueError("unseen property value")
                        act = mp[key]
                        out[m] = background(g) if act == "DEL" else act
                    return out
                if all(eq(apply(p["input"]), p["output"]) for p in pairs):
                    return [(f"object_map[{prop},conn{conn},{'col' if bycolor else 'all'}]", apply)]
    return []

def rule_object_map(task):
    r = rf_object_map(task["train"])
    return r[0][1] if r else None
