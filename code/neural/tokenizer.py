"""Serialize an ARC task into a flat token sequence for a small decoder-only transformer.

Vocab (18): 0-9 colors, 10 PAD, 11 NL(row end), 12 IN, 13 OUT, 14 QIN, 15 QOUT(answer start), 16 EOS, 17 BOS.
Sequence: BOS  [IN grid OUT grid]* (demos)  QIN grid QOUT  [grid EOS] (answer, train only).
Grid = rows of color tokens, each row terminated by NL. Structure tokens carry the 2D layout implicitly
(v1 uses 1D positions only; add 2D pos-emb later if needed). Knowledge-free: only ARC-train + augmentation.
"""
import numpy as np

PAD, NL, IN, OUT, QIN, QOUT, EOS, BOS = 10, 11, 12, 13, 14, 15, 16, 17
VOCAB = 18

def grid_tokens(g):
    toks = []
    for row in g:
        toks.extend(int(v) for v in row)
        toks.append(NL)
    return toks

def encode_task(demos, q_input, q_output=None, with_answer=False):
    """demos: list of {'input','output'} np grids. Returns token list (+ answer if with_answer)."""
    toks = [BOS]
    for d in demos:
        toks.append(IN); toks += grid_tokens(d["input"])
        toks.append(OUT); toks += grid_tokens(d["output"])
    toks.append(QIN); toks += grid_tokens(q_input)
    toks.append(QOUT)
    ans_start = len(toks)
    if with_answer and q_output is not None:
        toks += grid_tokens(q_output); toks.append(EOS)
    return toks, ans_start

def decode_grid(tokens):
    """Decode tokens generated after QOUT (up to EOS) into a grid. Robust to ragged rows."""
    rows, cur = [], []
    for t in tokens:
        t = int(t)
        if t == EOS or t == PAD or t == BOS:
            break
        if t == NL:
            if cur: rows.append(cur); cur = []
            continue
        if 0 <= t <= 9:
            cur.append(t)
        # ignore stray structure tokens
    if cur: rows.append(cur)
    rows = rows[:30]  # ARC grids are <=30 rows
    if not rows:
        return np.zeros((1, 1), dtype=np.int8)
    w = min(max(len(r) for r in rows), 30)  # <=30 cols
    if w == 0:
        return np.zeros((1, 1), dtype=np.int8)
    arr = np.zeros((len(rows), w), dtype=np.int8)
    for i, r in enumerate(rows):
        arr[i, :min(len(r), w)] = r[:w]
    return arr

def encode_train_example(demos, q_in, q_out, max_len=2048):
    """Full sequence + loss mask (loss only on answer tokens). Returns (ids, mask) or None if too long."""
    toks, ans_start = encode_task(demos, q_in, q_out, with_answer=True)
    if len(toks) > max_len:
        return None
    ids = np.array(toks, dtype=np.int64)
    mask = np.zeros(len(toks), dtype=np.float32)
    mask[ans_start:] = 1.0   # supervise only the answer region
    return ids, mask
