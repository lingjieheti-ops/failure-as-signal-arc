"""Inspect ARC-AGI-2 data: confirm schema + gather dataset statistics (paper material)."""
import json, glob, os, collections
from statistics import mean, median

DATA = r"D:/赚钱钱/Kaggle/ARC Prize 2026 - Paper Track/data/ARC-AGI-2/data"

def load_dir(split):
    tasks = {}
    for fp in glob.glob(os.path.join(DATA, split, "*.json")):
        tid = os.path.splitext(os.path.basename(fp))[0]
        with open(fp) as f:
            tasks[tid] = json.load(f)
    return tasks

for split in ("training", "evaluation"):
    T = load_dir(split)
    n_train_pairs, n_test, in_h, in_w, out_h, out_w, ncolors = [], [], [], [], [], [], []
    same_shape = 0
    keys_seen = set()
    for tid, t in T.items():
        keys_seen.update(t.keys())
        n_train_pairs.append(len(t["train"]))
        n_test.append(len(t["test"]))
        for pair in t["train"]:
            ih, iw = len(pair["input"]), len(pair["input"][0])
            oh, ow = len(pair["output"]), len(pair["output"][0])
            in_h.append(ih); in_w.append(iw); out_h.append(oh); out_w.append(ow)
            same_shape += int((ih, iw) == (oh, ow))
            cols = set(c for row in pair["input"]+pair["output"] for c in row)
            ncolors.append(len(cols))
    total_pairs = len(in_h)
    print(f"\n===== {split}: {len(T)} tasks =====")
    print(f"top-level keys present: {sorted(keys_seen)}")
    print(f"train pairs/task: min {min(n_train_pairs)} max {max(n_train_pairs)} mean {mean(n_train_pairs):.2f} median {median(n_train_pairs)}")
    print(f"test inputs/task: min {min(n_test)} max {max(n_test)} mean {mean(n_test):.2f}")
    print(f"input  H: {min(in_h)}-{max(in_h)} mean {mean(in_h):.1f} | W: {min(in_w)}-{max(in_w)} mean {mean(in_w):.1f}")
    print(f"output H: {min(out_h)}-{max(out_h)} mean {mean(out_h):.1f} | W: {min(out_w)}-{max(out_w)} mean {mean(out_w):.1f}")
    print(f"same input/output shape: {same_shape}/{total_pairs} pairs ({100*same_shape/total_pairs:.1f}%)")
    print(f"distinct colors/pair: min {min(ncolors)} max {max(ncolors)} mean {mean(ncolors):.2f}")

# show one task's raw schema
ev = load_dir("evaluation")
tid0 = sorted(ev)[0]
t0 = ev[tid0]
print(f"\n===== sample task {tid0} schema =====")
print("train[0].input  =", t0["train"][0]["input"])
print("train[0].output =", t0["train"][0]["output"])
print("test[0] keys    =", sorted(t0["test"][0].keys()))
