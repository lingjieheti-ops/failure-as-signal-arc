# ARC Prize 2026 — Paper Track: Winning Strategy

*Working backbone doc. Adapted from the SCI自动化启动 methodology to a Kaggle paper award.*
*Created 2026-06-09. Paper deadline: 2026-11-08 (~5 months).*

---

## 1. Objective & realistic target

- **Stated goal:** win (1st = $50K).
- **Honest target:** the prize structure is **not** winner-take-all. Beyond the $75K podium there is a **$375K "Outstanding Papers" pool for any paper scoring >4.5/5**. So the rational objective is **a paper that scores >4.5 across all six categories**. That both maximizes podium odds *and* unlocks the pool. We aim for the podium; we engineer for >4.5.
- **Bar to clear:** 2025 paper winners (TRM, SOAR, CompressARC) were serious research. We must produce a genuine, defensible novel contribution + a *real working leaderboard entry*. This is a research project, not a writing task.

## 2. The rubric, decoded → "winning paper profile"

Six categories, each 0–5, judged equally. To score >4.5 *everywhere*, the contribution must simultaneously satisfy:

| Category | What the judge asks | What maximizes it | Design implication |
|---|---|---|---|
| **Accuracy** | leaderboard score | a real, reproducible Kaggle entry with a non-trivial score | must build & submit; pick a method that *runs in the no-internet Kaggle notebook within budget* |
| **Universality** | generalizes beyond ARC? | a *mechanism/principle*, not an ARC-specific hack | frame as a general reasoning mechanism; show transfer / argue generality |
| **Progress** | raises anyone's chance of 85%? | credibly attacks the *real* bottleneck (compositional generalization) and/or the organizers' stated need | target the documented cliff, not an incremental hack |
| **Theory** | *why* it works, not just *how* | a principled account (MDL/compression, learning theory, info theory) + analysis | derive/justify, include ablations that test the *mechanism* |
| **Completeness** | thorough docs of the entry | full method, ablations, failure analysis, cost, reproducibility | open-source + appendix; report cost (only 11% of papers do!) |
| **Novelty** | novel vs public research | not a rehash of TTT / program synthesis / tiny-recursion | must do something the 2025 field did *not* |

**Cross-cutting "free points" most papers miss** (cheap, high-leverage):
- **Report cost/task** (only 11% of surveyed papers do; survey flags an 87% transparency gap).
- **Evaluate on the full set** (survey: <100-task evals inflate scores ~27pts) → our numbers are honest, which judges reward.
- **Open-source cleanly** (required anyway; raises Completeness).

## 3. Competitive landscape (what's saturated vs open)

**Saturated / done (low Novelty if repeated):**
- Test-time training / fine-tuning (NVARC, MindsAI, ARChitects) — every >70% ARC-1 system uses it.
- LLM-guided program synthesis with refinement loops (SOAR, Berman, Pang) — "refinement is intelligence" was *the* 2025 theme.
- Tiny recursive networks (TRM) — 1st place 2025; copying it = derivative.
- MDL/compression zero-pretrain (CompressARC) — 3rd place 2025; needs a real twist to be novel.

**Where we cannot compete:** raw scale. NVARC used 266K synthetic puzzles; ARChitects 39h on 8×H100; commercial systems are uncapped. On a single RTX 5080 we **win on ideas, not compute** — which is exactly what the rubric rewards (Theory/Novelty/Progress > Accuracy alone, 5:1).

## 4. The exploitable gaps (ranked by rubric leverage)

From the official technical report + living survey, explicitly stated as open:

1. **Compositional generalization cliff** — 70–80% on ARC-1 → 20–30% on ARC-2, *no paradigm immune*. "No current architecture implements explicit mechanisms for hierarchical reasoning." → highest Progress.
2. **Catastrophic vs graceful failure** — AI drops 75%→~0 with *no partial solutions, no diagnostics of missing primitives*; humans degrade gracefully. → highest Novelty (almost untouched).
3. **Knowledge vs reasoning separation** — organizers *explicitly* wrote: "we still need... methods to separate knowledge and reasoning." → answering a stated need = high Progress.
4. **Scaling ≠ compression / efficiency** — "more with more" vs human "more with less"; 7-orders-of-magnitude energy gap. → Theory + Universality.
5. **Self-grounding primitives** — no autonomous primitive discovery with confidence/verification. → Novelty + Theory.
6. **ARC-AGI-3 interactive** — 8× human gap, near-total static→interactive collapse; field essentially empty. → highest Novelty ceiling, highest build risk.
7. **Evaluation rigor** — transparency/cost/contamination. → a meta-angle to *fold into* any paper (not a standalone bet, because the track requires a working entry).

## 5. Candidate research directions

Each scored 1–5 for rubric-fit and feasibility-on-5080. (Detailed in §6; user picks in the kickoff question.)

### A — Graceful Compositional Reasoning *(recommended primary bet)*
**Idea:** a **knowledge-free** (zero ARC-pretraining; learned per-task at test time) solver that explicitly **induces a *hierarchy* of reusable primitives** (compositional, not flat search), and crucially **degrades gracefully**: when it can't fully solve a task it outputs a **partial solution** + a **"primitive-gap" diagnostic** (which abstraction it's missing). Grounded in **MDL/compression** theory: the correct program is the one that maximally compresses the train pairs; composition = hierarchical compression.
- Hits **all six**: real entry (Accuracy) · general mechanism (Universality) · attacks the #1 cliff + catastrophic-failure gap (Progress) · MDL "why" (Theory) · we document fully (Completeness) · graceful degradation + primitive-gap diagnostics + knowledge-free hierarchical induction are jointly **novel** (survey flags all three as missing).
- Bonus: knowledge-free ⇒ also answers gap #3 (knowledge/reasoning separation).
- Feasibility: high (CompressARC-class compute, 5080-friendly). Risk: medium accuracy, but Accuracy is 1/6 and modest scores have placed.

### B — Compression-first reasoning (MDL++)
**Idea:** push the CompressARC/MDL line with a genuine advance — e.g., *amortized* or *compositional* compression so it's no longer 20-min/task and handles multi-step transforms.
- Theory/Universality: 5. Novelty: 3 (line exists). Feasibility: very high (76K params).

### C — ARC-AGI-3 interactive agent *(high-risk / high-ceiling)*
**Idea:** a first-principles agent for the new interactive benchmark (exploration + world-model + memory), with an analysis of *why* static methods collapse.
- Novelty/Progress ceiling: 5 (field empty). Feasibility: low–medium (RL-ish, sparse reward, long horizon; tight Jun 30/Sep 30 milestones). Need to confirm the paper track accepts ARC-AGI-3 entries.

### D — Knowledge–reasoning separation (diagnostic + method)
**Idea:** operationalize the organizers' stated need: a method/diagnostic that *measures and removes* knowledge contamination and isolates pure reasoning (builds on the zero-pretrain line).
- Progress: 5 (stated need). Novelty: 4. Feasibility: medium. Risk: "working entry" must still score.

> A, B, D share a **knowledge-free / compression** spine and can partially merge. C is a different bet (different benchmark, code, risk).

## 6. Recommended primary bet — rationale

**Direction A**, because it is the *only* option that plausibly scores >4.5 on **all six** categories at once on our compute:
- It attacks the field's #1 documented blocker (Progress) with a mechanism nobody has combined (Novelty), justified by a real theory (Theory), as a knowledge-free general principle (Universality), via an entry we can actually run and fully document on a 5080 (Accuracy + Completeness).
- The "graceful degradation + primitive-gap diagnostics" framing is a genuinely fresh lens: it reframes ARC progress from "raise top-1 accuracy" to "make failure informative," which is both novel and a real path toward 85% (diagnostics → targeted primitive learning).
- Fallback safety: even if accuracy is modest, the Theory/Novelty/Progress/Universality/Completeness scores carry it into the >4.5 pool — the same profile that placed CompressARC (3rd) and TRM (1st) despite single-digit ARC-2 scores.

## 7. Phased roadmap to Nov 8 (SCI-adapted)

| Phase | Window | Output | Gate |
|---|---|---|---|
| **0. Strategy + direction lock** | now | this doc + user pick | ← we are here |
| **1. Deep landscape + idea crystallization** | Jun | read TRM/SOAR/CompressARC source, confirm Kaggle notebook constraints, sharpen the precise novel claim + theory | clear 1-sentence contribution + testable hypothesis |
| **2. Prototype + baseline** | Jun–Jul | minimal working solver on ARC-AGI-2 public eval; reproduce a known baseline first | a number on the public set |
| **3. Build the contribution** | Jul–Sep | implement hierarchy + graceful-degradation + diagnostics; ablations | mechanism beats baseline on the metric we care about |
| **4. Real Kaggle submission** | Sep–Oct | reproducible no-internet notebook; leaderboard score; cost/task | **hard gate: working entry on leaderboard** |
| **5. Write paper (SCI pipeline)** | Oct | Writing-Rationale-Matrix → closed-book draft → figures (1200dpi) → full ablations/cost/failure analysis | every rubric category explicitly addressed |
| **6. Adversarial review + polish** | Oct–early Nov | 7-agent peer review vs the 6-category rubric; ai-review hardening; reproducibility check; open-source repo | self-scored >4.5 each category |
| **7. Submit** | by Nov 8 | paper + Kaggle entry + open-source repo | — |

## 8. Open items to confirm during Phase 1
- Exact Kaggle ARC-AGI-2 2026 notebook limits (GPU, runtime, no-internet) — sets the compute envelope for the entry.
- Whether the paper track accepts an **ARC-AGI-3** entry as the "working submission" (affects Direction C).
- Paper format/length/template requirements (not on the rubric page; check Kaggle rules + 2025 examples).
- Any 2026 entries already public (avoid collision; the comp launched Mar 25).

## 9. Decisions needed from user (kickoff)
1. **Direction** (A recommended / B / C / D).
2. **Compute & involvement reality** — only the 5080, or extra cloud budget? Can you run Kaggle GPU notebooks + actively co-run training? This sets feasibility.
3. **Risk appetite** — swing for 1st (accept high variance, e.g. C) vs engineer for the >4.5 pool (A).
