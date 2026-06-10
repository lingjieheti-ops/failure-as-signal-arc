# Neural route — unified hybrid design (integrates all prior work)

Goal: lift accuracy to a respectable (TRM-ish, ~5-8% ARC-2) level WITHOUT discarding the knowledge-free
symbolic + diagnostic contributions. The trick: make them ONE system.

## Architecture: "Interpretable where possible, neural where necessary"
For each task:
1. **Symbolic first (exact + interpretable):** run the knowledge-free MDL searcher (`search.py`).
   If it finds a program with ZERO train residual → use it. attempt is exact, human-readable, knowledge-free.
2. **Neural fallback (accuracy):** else, a tiny recursive solver predicts the grid (the TRM-style engine).
3. **Diagnostic routes & explains (novelty):** the MDL diagnostic decides confidence/routing, emits the
   partial-credit (C2) and the missing-primitive signature (C3) for whatever neither solves.
4. **Frontier map structures evaluation:** report accuracy per the 7 meta-families (E7) — ties results to
   the diagnostic framework; shows WHERE the neural net helps vs where the gap remains.

This keeps C1/C2/C3 + frontier map intact and adds a neural accuracy engine + a routing story. Novel:
nobody combines knowledge-free symbolic + tiny-neural + an MDL diagnostic that names the residual.

## The tiny recursive neural solver (engine)
- Repr: grid -> token grid (10 colors + pad + border), 2D rotary/learned positional enc; task = a few
  (input,output) demo pairs + a test input, encoded as a sequence/stack the model attends over.
- Model: small (~5-10M param) recursive transformer (deep supervision: refine the answer over K steps),
  in the spirit of TRM (2510.04871) but our own clean impl. Fits 5080 easily (tiny).
- Train: 1000 ARC-2 train tasks x heavy augmentation (D4 x color-perm x example-shuffle ≈ 1000x) ->
  millions of synthetic examples. AdamW, mixed precision (bf16 on Blackwell).
- Test time: augmentation-vote (predict over D4/color-perm variants, majority) — the proven +acc trick.
  pass@2 = top-2 voted candidates.

## Build order (once GPU verified)
1. `tokenizer.py` — grid<->tokens, task serialization, output decode (+ predict output shape).
2. `augment.py` — D4 x color-permutation x pair-shuffle (knowledge-free augmentations only).
3. `model.py` — tiny recursive transformer + deep-supervision loss.
4. `train.py` — training loop (bf16, checkpointing) on the 5080.
5. `infer.py` — augmentation-vote inference -> attempt_1/2.
6. `hybrid.py` — symbolic-exact-first, neural-fallback, diagnostic-routed; writes submission.json.
7. measure eval pass@2 + per-frontier-family breakdown. Decide go/no-go vs all-in-concepts.

## Go / no-go (the user's "不行的话就全押概念")
- GO if hybrid eval pass@2 >= ~5% (meaningfully > symbolic ~1-2%) and trains/infers within Kaggle 12h.
- NO-GO (fall back to all-in concepts) if: GPU/Blackwell unstable, OOM, train too slow, or acc stays ~symbolic.

## Kaggle constraint check
Inference must run offline in <12h on L4x4. Tiny model + aug-vote on 120 tasks = minutes. Training is done
OFFLINE beforehand; we ship the trained weights in the notebook (allowed). Symbolic half is CPU, fast.
