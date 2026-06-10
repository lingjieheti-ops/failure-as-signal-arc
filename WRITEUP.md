# Failure as Signal: Knowledge-Free Compositional Reasoning that Maps the Path to ARC-AGI

**Author:** Si Fan

**Paper (PDF):** https://github.com/lingjieheti-ops/failure-as-signal-arc/blob/master/paper/main.pdf
**Open-source code (required):** https://github.com/lingjieheti-ops/failure-as-signal-arc
**Linked Kaggle code submission (ARC-AGI-2):** https://www.kaggle.com/code/yuanyezhiyin2/failure-as-signal-entry

## Abstract
On ARC-AGI-2, every published method falls off a *compositional cliff*: scores of 70–80% on ARC-AGI-1
collapse to 20–30%, with no paradigm immune. We argue the bottleneck is not only that solvers fail, but that
they fail *uninformatively* — producing a wrong grid and no account of what is missing. We reframe the
problem: a solver should treat **failure as signal**. We present a knowledge-free (no pretraining on ARC, no
neural network in the submitted entry) solver that searches a typed, compositional hypothesis space under a
hierarchical MDL objective, and on every task returns (i) its best *partial* solution and (ii) a *typed
diagnosis of the missing abstraction*. Under a two-part MDL code, the incompressible residual of the best
program localizes the gap and certifies that re-adding the missing primitives drives the residual to zero
(Theorem 1). We turn this into a measured **gap-closure loop** run for three iterations on ARC-AGI-2, where
it acts on its own diagnoses; we probe generality on two non-ARC domains (an authored sequence domain — 92%
exact, 100% ablation closure, 62% signature naming — and the externally-defined space of all 256 elementary
cellular automata — exact recovery of the library-expressible family, graceful degradation in the
non-realizable regime at +0.34 cell-accuracy over copy-input, and residual size predicting single-primitive
closability at AUC 0.76). We also contribute a **frontier map**: an unsupervised clustering (with measured
stability) of the ~96% of ARC-AGI-2 evaluation tasks that no single-family detector can name, into seven
structural meta-families that prioritize what the field should build next.

## Honesty, up front
- The submitted entry is purely symbolic, CPU-only, offline; it solves **1/120 (0.83%)** of the public
  evaluation set. The rules state the code submission "need not achieve a high score"; the paper's value is
  the mechanism, theory, generality, and roadmap.
- We disclose that the symmetry-repair operator was generalized after inspecting one evaluation task; the
  frozen pre-inspection library scores 0/120. All later capabilities were derived from **training-set
  residuals only**, with **two pre-registered clean evaluation runs** — both left the score unchanged. Three
  consecutive directed iterations lift training coverage monotonically yet leave evaluation at exactly 1/120:
  evaluation tasks are structurally novel compositions, not harder instances of training families. We believe
  this adverse result is itself one of the paper's most useful findings.
- A 40M-parameter from-scratch transformer we explored is reported as a quantified negative result (0.916
  token accuracy, zero exact matches in or out of distribution; 0.916^N ≈ 0 for N≈100-cell grids).
- Every number in the paper traces to the public experiment log (`experiments/RESULTS.md`).

## Why this advances the path to 85%
1. **A measured, interpretable self-extension loop** — diagnose the residual, name the missing primitive,
   add it, observe transfer (or its failure). Every capability is named and human-auditable.
2. **A theorem giving the loop its license** — the residual provably localizes and (in the singleton-gap
   regime) certifies closability; validated under controlled ablation (135/135) and from outside on ECA.
3. **A prioritized frontier map of the unsolved evaluation set** (released per-task), telling the field which
   abstraction families matter most — with the adverse periodicity datum already sharpening it to
   "compositional periodicity."
4. **A partial-credit metric** making sub-pass@2 progress measurable (shape-commit rate ×
   conditional cell fidelity), where the binary leaderboard metric hides it.

## Reproduce
```
git clone https://github.com/lingjieheti-ops/failure-as-signal-arc
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 data/ARC-AGI-2
python code/submit_symbolic.py --split evaluation   # the entry: 0.83%, format-valid, ~4 min CPU
# every paper experiment: see README.md (one command each)
```
