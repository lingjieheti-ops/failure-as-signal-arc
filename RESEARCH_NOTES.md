# Research Notes — the contribution (Phase 1 crystallization)

*Locked direction: A — Graceful Compositional Reasoning. Compute: RTX 5080 + Kaggle GPU. Target: >4.5 in every rubric category.*
*This doc is the intellectual core + the seed of the paper's Writing-Rationale-Matrix.*

---

## 0. Working title

**"Failure as Signal: Knowledge-Free Compositional Reasoning that Maps the Path to ARC-AGI"**
*(alt: "When a Solver Knows What It's Missing: Residual-MDL Diagnostics for ARC-AGI")*

## 1. One-sentence thesis

A **knowledge-free, anytime** solver for ARC-AGI-2 that searches a **typed compositional** hypothesis space under a **hierarchical MDL** objective, and on *every* task returns (i) its **best partial solution** and (ii) a **typed diagnosis of the missing abstraction** — turning the field's catastrophic-failure cliff into a measurable, navigable map toward 85%.

## 2. The wedge (why this is the right gap)

- The #1 documented blocker is the **compositional generalization cliff**: 70–80% on ARC-1 → 20–30% on ARC-2, *no paradigm immune* (survey 2603.13372; tech report 2601.10904).
- **CompressARC** (Liao & Gu, knowledge-free MDL, 2512.06104) proves compression-at-test-time works for *perceptual/local* abstraction but **explicitly cannot** do: counting, repeated/multi-step operations, translation/rotation/reflection/duplication, topological connectivity, long-range patterns. → A *flat* per-pixel MDL decoder has **no hypothesis space for composition**. This is the cliff, instantiated structurally.
- Current systems fail **catastrophically** (75%→~0, no partial output, no diagnosis) where humans **degrade gracefully** (partial success + systematic errors that reveal strategy). Survey flags this as essentially **untouched**.
- Organizers explicitly asked for **"methods to separate knowledge and reasoning."** Knowledge-free ⇒ answers this directly.

## 3. Three novel claims (the contribution)

**C1 — Anytime knowledge-free hierarchical induction.**
Per-task (no pretraining, no LLM, no ARC priors), search a *typed, compositional* hypothesis space with a **hierarchical MDL** prior where *reuse shortens the code* — so compression pressure *prefers compositional* explanations. This is the structural fix CompressARC lacks. **Anytime**: maintains a current-best at any budget ⇒ fits the 12h/120-task Kaggle envelope and degrades under budget.

**C2 — Graceful degradation via partial-program execution.**
When no program explains all train pairs / all cells, execute the **best partial program** and emit its prediction as the attempt. Report a new **partial-credit metric** (cell- and region-level), converting binary 0s into informative partials. (Leaderboard Accuracy stays pass@2 exact-match; partial-credit is our analysis metric.)

**C3 — Primitive-gap diagnostics (the headline novelty).**
Define **missing primitive = the incompressible residual** under the current primitive set (the part of the transform with no shorter encoding). Localize + type it into a diagnosis ("missing: object-count / topological-fill / periodic-extension / color-bijection…"). **Validate** that (a) diagnoses correlate with ground-truth task tags, and (b) *adding the diagnosed primitive closes the gap* on held-out tasks — a concrete, measurable mechanism toward 85% (Progress, operationalized).

## 4. Theory — *why* it works (the MDL backbone → the "Theory" rubric category)

1. **Occam/Solomonoff:** the shortest program reproducing the train pairs is the best generalizer ⇒ MDL objective is principled, not heuristic.
2. **Composition is provably cheaper:** a two-part code for a program with reusable subroutines is strictly shorter than the flat concatenation of operations ⇒ a hierarchical MDL prior *provably prefers* compositional solutions. (This is exactly the expressivity a flat decoder can't capture.)
3. **Residual = missing primitive (the key theorem-shaped statement):** under the current library L, the description length decomposes as DL(program | L) + DL(residual | program). The residual is, by definition, the component with **no shorter encoding in L** — i.e., the *signature of a primitive absent from L*. Diagnostics = localizing & typing this residual. This gives a *principled definition* of "what the solver is missing," which is what makes C3 a theory contribution, not a heuristic.

## 5. Differentiation (must-cite related work + how we differ)

| Work (arXiv) | What it does | How we differ |
|---|---|---|
| CompressARC — Liao & Gu (2512.06104) | knowledge-free **flat** per-pixel MDL, test-time | we add **compositional hypothesis space** + **graceful partial output** + **diagnostics**; fix its stated limitations |
| TRM — Jolicoeur-Martineau (2510.04871) | tiny **recursive** net, **trained on ARC** (knowledge-bound) | we are **knowledge-free**; failure-as-signal, not just answer refinement |
| SOAR — Pourcel et al. (2507.14172) | **LLM** evolutionary program synthesis + self-FT | we use **no LLM**, no pretraining; runs on a 5080 |
| Neurally-guided induction — (2411.17708) | MDL + **residual-guided search** | residual guides *search* there; we make residual a **typed diagnosis of missing concepts** + graceful output |
| Compositional-Gap affinity (2512.07109) | *diagnoses* gap in **transformers** (cell vs grid acc) | we adopt the cell-vs-grid framing but build a **solver** that *acts on* the diagnosis (closes gaps) |
| DreamCoder — Ellis et al. (2006.08381) | **cross-task** library learning (knowledge-accumulating) | we are **per-task, knowledge-free, anytime**; no learned library, no base-DSL training |
| VSA for ARC (2511.08747) | vector-symbolic abstraction | different substrate; we keep symbolic+MDL for interpretable diagnostics |

**Novelty statement (defensible):** the *combination* — knowledge-free **per-task** hierarchical-MDL composition + **graceful partial-solution output** + **residual-as-missing-primitive typed diagnostics** with a **closes-the-gap demonstration** — is not present in any single prior work.

## 6. Substrate = the first Phase-2 experiment (NOT yet decided)

The hypothesis space + MDL engine could be:
- **(S) Symbolic** typed-DSL search w/ MDL = program length. Pros: naturally compositional, fast (CPU/GPU, fits 12h easily), interpretable diagnostics. Cons: combinatorial explosion (the known DSL-search ceiling) → needs strong MDL pruning + the anytime mechanism.
- **(N) Neural** CompressARC-style but with compositional/recurrent structure. Pros: differentiable, elegant. Cons: slow (~20min/task), the limitations are architectural.
- **(H) Hybrid** knowledge-free perceptual front-end (objects/relations via untrained equivariant net or classical CV) → symbolic compositional induction with hierarchical MDL. **Current bet** (best composition + speed + interpretable residual), but most to build.

**Decision rule:** Phase 2 prototypes S and H minimally on the public eval; pick by (compositional reach × speed-within-12h × diagnostic interpretability). Lock by end of Phase 2.

## 7. Metrics we will report (Completeness + honesty = free rubric points)

- **Leaderboard pass@2 exact-match** (the Accuracy category; full eval set, not a cherry-picked subset — survey flags <100-task inflation).
- **Cost/task** (only 11% of papers report it — easy Completeness win).
- **Partial-credit** (cell-level, region-level) — our graceful-degradation metric (C2).
- **Diagnostic accuracy** vs ground-truth task tags (C3a).
- **Gap-closure rate**: % of failed tasks solved after adding the diagnosed primitive (C3b → Progress).
- Ablations isolating each mechanism (hierarchy on/off, graceful on/off, diagnostics on/off).

## 8. De-risked build plan (hard gate FIRST)

1. **Secure the hard gate early:** get ARC-AGI-2 data → trivial baseline (identity / tiling / most-frequent-color) → a *valid `submission.json`* runs end-to-end in a Kaggle notebook (no internet, <12h, pass@2). Banks Accuracy>0 + Completeness pathway before any research risk.
2. Reproduce a *simple known* solver (e.g., a small DSL search) to calibrate.
3. Build C1 (hierarchical MDL composition) → measure vs baseline.
4. Add C2 (graceful partial output) + C3 (diagnostics) → validate (3)(4)(5) metrics.
5. Lock substrate; scale; final reproducible notebook + open-source repo.
6. SCI writing pipeline → 7-agent rubric review → submit by Nov 8.

## 9. Open items (resolve in Phase 2)
- Paper format/length/template — not on the rubric page; Kaggle page is JS-rendered (use Chrome MCP or infer from 2025 arxiv papers; target NeurIPS/arxiv ~8–12pp + appendix).
- Exact Kaggle 2026 ARC-AGI-2 quota (L4×4, 12h confirmed; verify GPU-hour quota).
- Speed budget: 120 tasks / 12h ⇒ ≤6 min/task on L4×4 (CompressARC's 20min/task does NOT fit as-is → anytime/early-stop or symbolic speed needed).

## 10. Must-cite anchor set
2603.13372 (survey) · 2601.10904 (2025 tech report) · 2505.11831 (ARC-AGI-2, Chollet) · 2512.06104 (CompressARC) · 2510.04871 (TRM) · 2507.14172 (SOAR) · 2411.17708 (neurally-guided induction) · 2512.07109 (compositional-gap diagnostic) · 2006.08381 (DreamCoder) · 2511.08747 (VSA) · 2507.15877 (OOD: exec-guided NPS vs TTFT).
