"""Generate the paper's three key figures (vector PDF, academic style, no in-figure titles)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})
BLUE, GRAY, ORANGE = "#2E5C8A", "#9aa0a6", "#C55A11"
OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

# --- Fig 1: gap-closure loop (training), two iterations ---
fam = ["recolor", "symmetry", "dedup", "scale", "panel", "periodic", "object\ncount"]
diag = [5, 5, 4, 4, 3, 1, 31]
clos = [4, 4, 4, 3, 1, 1, 4]
x = np.arange(len(fam))
fig, ax = plt.subplots(figsize=(5.4, 2.6))
ax.bar(x - 0.2, diag, 0.4, label="diagnosed", color=GRAY)
ax.bar(x + 0.2, clos, 0.4, label="closed by named primitive", color=BLUE)
ax.set_xticks(x); ax.set_xticklabels(fam, fontsize=8)
ax.set_ylabel("# training tasks"); ax.legend(frameon=False, fontsize=8)
ax.annotate("flagged by iter. 1 -> built in iters. 2-3:" + chr(10) + "4/31 closed; family needs composition",
            xy=(6.2, 31), xytext=(2.9, 25.5),
            fontsize=7.5, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
fig.savefig(os.path.join(OUT, "fig_gapclosure.pdf"))
plt.close(fig)

# --- Fig 2: frontier map (eval cluster sizes) ---
labels = ["periodic-texture edits", "simple-palette transforms", "extraction / selection",
          "dense multi-color edits", "symmetrization", "tiling / expansion", "many-object"]
sizes = [30.0, 25.8, 22.5, 13.3, 5.8, 1.7, 0.8]
fig, ax = plt.subplots(figsize=(5.4, 2.6))
y = np.arange(len(labels))[::-1]
ax.barh(y, sizes, color=BLUE)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("% of ARC-AGI-2 evaluation set")
for yi, s in zip(y, sizes):
    ax.text(s + 0.4, yi, f"{s:.1f}", va="center", fontsize=7.5)
fig.savefig(os.path.join(OUT, "fig_frontier.pdf"))
plt.close(fig)

# --- Fig 3: generality ablation (sequence domain) ---
prims = ["reverse", "repeat2", "tail", "dedup", "rot_r", "head", "sort_desc",
         "sort_asc", "dec", "inc", "double", "rot_l"]
cat = [100, 100, 100, 93, 88, 86, 83, 40, 36, 27, 13, 0]
closed = [100] * len(prims)
x = np.arange(len(prims))
fig, ax = plt.subplots(figsize=(5.8, 2.6))
ax.bar(x - 0.2, closed, 0.4, label="gap closed by re-add", color=BLUE)
ax.bar(x + 0.2, cat, 0.4, label="missing-category named", color=ORANGE)
ax.set_xticks(x); ax.set_xticklabels(prims, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("%"); ax.set_ylim(0, 108); ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig(os.path.join(OUT, "fig_generality.pdf"))
plt.close(fig)
print("wrote figs:", os.listdir(OUT))
