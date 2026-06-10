"""Inference: autoregressive greedy decode + optional augmentation-vote -> attempt_1/2 -> submission.
v1 uses simple per-sequence greedy generation (no KV cache); fast enough since trained outputs early-stop
at EOS. Falls back to copy-input when the prompt won't fit."""
import sys, os, argparse, time
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
import arc_core as ac
from model import GPT
from tokenizer import encode_task, decode_grid, EOS, VOCAB
from augment import augment_task, invert_grid

@torch.no_grad()
def gen(model, prompt, dev, max_new):
    ids = torch.tensor(prompt, dtype=torch.long, device=dev)[None]
    out = []
    for _ in range(max_new):
        if ids.shape[1] >= model.max_len:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt = int(model(ids)[0, -1].float().argmax())
        if nxt == EOS:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], 1)
    return out

def predict(model, demos, q_in, dev, max_len):
    prompt, _ = encode_task(demos, q_in, with_answer=False)
    if len(prompt) >= max_len - 16:
        return q_in.copy()
    toks = gen(model, prompt, dev, max_new=min(1024, max_len - len(prompt)))
    return decode_grid(toks)

def solve_task(model, task, dev, max_len, n_aug=3):
    preds = []
    for ti, tp in enumerate(task["test"]):
        cands = [predict(model, task["train"], tp["input"], dev, max_len)]
        rng = np.random.default_rng(1234 + ti)
        for _ in range(n_aug):
            aug, (d4i, perm) = augment_task({"train": task["train"], "test": [tp]}, rng)
            g = predict(model, aug["train"], aug["test"][0]["input"], dev, max_len)
            cands.append(invert_grid(g, d4i, perm))
        key = lambda a: (a.shape, a.tobytes())
        kg = {key(x): x for x in cands}
        c = Counter(key(x) for x in cands)
        ranked = [kg[k] for k, _ in c.most_common()]
        a1 = ranked[0]; a2 = ranked[1] if len(ranked) > 1 else ranked[0]
        preds.append({"attempt_1": ac.to_ll(a1), "attempt_2": ac.to_ll(a2)})
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/gpt.pt")
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_aug", type=int, default=3)
    ap.add_argument("--out", default="experiments/sub_neural.json")
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev)
    cfg = ck["cfg"]
    model = GPT(VOCAB, cfg["d"], cfg["heads"], cfg["layers"], cfg["max_len"]).to(dev).eval()
    model.load_state_dict(ck["sd"])
    tasks = ac.load_tasks(a.split)
    items = sorted(tasks.items())
    if a.limit: items = items[:a.limit]
    sub = {}; t0 = time.time()
    for i, (tid, t) in enumerate(items):
        sub[tid] = solve_task(model, t, dev, cfg["max_len"], a.n_aug)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(items)} ({(time.time()-t0)/(i+1):.1f}s/task)", flush=True)
    ac.save_submission(sub, a.out)
    m = ac.evaluate(sub, {k: tasks[k] for k, _ in items})
    print(f"=== neural [{a.split}] {len(items)} tasks, {time.time()-t0:.0f}s ===")
    print(f"output_pass@2 {m['output_pass@2']*100:.2f}%  task_pass@2 {m['task_pass@2']*100:.2f}%  -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
