# Submission checklist — ARC Prize 2026 Paper Track

Everything is built and validated. The actual submit needs **your Kaggle account + identity** (I can't do that
step). Here is exactly what to do.

## 0. One thing only you can fill
- `paper/main.tex` line ~17: replace `[Author Name]` / `[email]` with your name and email.

## 1. Build the paper PDF
```bash
cd paper && latexmk -pdf main.tex      # -> paper/main.pdf  (10 pages, already compiles clean)
```
(Optional: upload to arXiv and use that link in the writeup.)

## 2. Create the ARC-AGI-2 code submission (the "working entry" the paper must link to)
The paper track requires a linked Kaggle code submission. Ours is **purely symbolic, offline, CPU, no model**:
1. Go to the **ARC Prize 2026 — ARC-AGI-2** Kaggle competition, "New Notebook."
2. Add the `code/` folder of this repo as a notebook utility-script / dataset (no internet needed).
3. In the notebook, run:
   ```python
   !python code/kaggle_submit.py          # auto-finds /kaggle/input test challenges -> /kaggle/working/submission.json
   ```
   It finishes in ~minutes (CPU), well inside the 12h limit, and writes a **format-valid** `submission.json`.
4. Submit. Expected score: low single digits (≈0.8% — that is fine and expected; see the paper).
   - Verify internet is OFF (required). The entry uses no internet and no GPU.

## 3. Create the Paper Track writeup
1. Go to **ARC Prize 2026 — Paper Track**, "New Writeup."
2. Attach / link `paper/main.pdf` (or the arXiv link).
3. Link the **ARC-AGI-2 code submission** from step 2 (this is the required "corresponding Kaggle submission").
4. Link the **open-source repo** (step 4).
5. Submit before **2026-11-08**.

## 4. Open-source the repo (required for eligibility)
Push this whole folder to a public GitHub repo (it has a top-level `README.md`). Make sure `data/ARC-AGI-2/`
is either included or its clone command is in the README (it is).

## What's in the box (all done, all honest)
- `paper/main.pdf` — 10-page paper: failure-as-signal framing, MDL theorem (gap-closability), frontier map,
  generality + controlled check, honest ARC-AGI-2 results, candid limitations. Survived two adversarial reviews.
- `code/kaggle_submit.py` — the offline, knowledge-free symbolic entry (validated end-to-end, 0 format errors).
- `experiments/submission_symbolic.json` — a ready, format-valid submission on the public eval set (0.83%).
- `experiments/RESULTS.md` — every number in the paper traces here.
- `README.md` — reproduce commands.

## Honest expectation (so there are no surprises)
The realistic target is the **>4.5 Outstanding-Papers pool** ($375K, not winner-take-all), on the strength of
the ideas (Novelty / Theory / Progress / Universality / Completeness) — **not** the leaderboard, which we
cannot win on a single machine. Accuracy is ~0.8% and we report it without spin; the rules state the entry
"need not achieve a high score." 1st place specifically is not realistic on this hardware; a pool placement is
a genuine, honest shot.
