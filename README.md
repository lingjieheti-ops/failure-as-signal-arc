# Failure as Signal — Knowledge-Free Compositional Reasoning for ARC-AGI-2

ARC Prize 2026 **Paper Track** entry. A knowledge-free (no ARC pretraining, no LLM) solver that treats
**failure as signal**: on every task it returns its best *partial* solution plus a *typed diagnosis of the
missing abstraction*, grounded in an MDL theorem (the incompressible residual **is** the missing primitive).
Honest about accuracy on a single consumer GPU; the contribution is mechanism, theory, generality, and a
prioritized roadmap.

Paper: `paper/main.pdf` (build: `cd paper && latexmk -pdf main.tex`).

## What's here
| path | what |
|---|---|
| `code/arc_core.py` | ARC-AGI-2 IO + faithful pass@2 evaluator + submission format |
| `code/dsl.py` | ~22 **knowledge-free, generic** primitives (geometry, objects, axis-search symmetry repair, soft periodicity) |
| `code/search.py` | anytime **MDL compositional search** (C1) |
| `code/solver_baseline.py` | depth-1 transform bank (motivating baseline) |
| `code/diagnose.py` | structural **diagnostic taxonomy** (C3 detectors) |
| `code/gap_closure.py` | **diagnose → add named primitive → close gap** loop (C3b) |
| `code/counting.py` | iteration-2 cardinality families (count→grid, select-by-count, learned count→color) |
| `code/partial_credit.py`, `code/run_graceful.py` | **graceful degradation** + partial-credit metric (C2) |
| `code/frontier.py` | **frontier map**: cluster the unsolved 95% into meta-families |
| `code/generality/seq_domain.py` | generality domain 1 (authored sequences): ablation validation of the theorem |
| `code/generality/ca_domain.py` | generality domain 2 (**external**: 256 elementary CA): non-realizable regime + residual→closability |
| `code/neural/` | tiny GPT fallback (tokenizer, augment, model, train, infer) + **`hybrid.py`** |
| `code/kaggle_submit.py` | offline Kaggle entry: challenges → hybrid → `submission.json` |
| `code/validate_submission.py` | Kaggle-format validator (hard-gate) |
| `experiments/RESULTS.md` | full, honest experiment log (every number in the paper traces here) |
| `paper/` | LaTeX source + figures of the paper |

## Reproduce the key results (CPU, no GPU needed)
```bash
# data (public ARC-AGI-2)
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 data/ARC-AGI-2

python code/run_eval.py evaluation        # depth-1 baseline (0% eval, 1.6% train) — the cliff
python code/test_dsl.py evaluation        # generic single-primitive reach (27 prims; train 3.0%)
python code/diagnose.py evaluation        # diagnostic taxonomy (names ~3% of eval)
python code/gap_closure.py training       # gap-closure loop, 2 iterations (incl. counting; union 20/1000)
python code/partial_credit.py evaluation  # C2: graceful, 97.6% non-catastrophic
python code/frontier.py evaluation 7      # the frontier map (7 meta-families)
python code/frontier.py stability evaluation 7   # cluster-stability (ARI across seeds and k)
python code/generality/seq_domain.py      # domain 1 (authored): 92% solved + 100% ablation closure
python code/generality/ca_domain.py       # domain 2 (external, 256 ECA rules): 43/256 exact;
                                          #   non-realizable graceful +0.342; residual->closability AUC 0.76
```

## The official entry (symbolic, CPU-only, offline)
```bash
python code/submit_symbolic.py --split evaluation                 # full eval: 0.83% pass@2, VALID format
python code/kaggle_submit.py --challenges <test_challenges.json>  # the offline Kaggle entry -> submission.json
```
No GPU, no internet, ~2.5 min for 120 tasks.

## The transformer we explored (negative result, not submitted)
```bash
python code/neural/train.py --steps 40000 ...   # tiny GPT from scratch (Blackwell sm_120 / cu128)
python code/neural/infer.py --ckpt experiments/gpt_24k.pt --split evaluation
```
Reached 0.916 token accuracy yet 0 exact matches in or out of distribution (0.916^N ≈ 0 for N≈100-cell
grids) — reported transparently in the paper's Limitations; checkpoints kept for reproducibility.

## Honest results (ARC-AGI-2 public eval, pass@2)
Knowledge-free single-GPU accuracy is low single digits (the leaderboard is topped near 24% with industrial
compute). Per the rules the entry "need not achieve a high score." The value is the framework: a measured
gap-closure loop (75–100% per named family), graceful non-catastrophic failure (97.6%), an MDL theorem, a
generality demo (92% on a non-ARC domain with 100% gap-closure), and a frontier map / roadmap. See
`experiments/RESULTS.md` for everything, reported without embellishment.

## License
Code released open-source for ARC Prize eligibility. Data: ARC-AGI-2 (see its repo's license).
