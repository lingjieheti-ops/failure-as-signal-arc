"""Streaming augmented training examples: sample task -> D4xcolor augment -> leave-one-out query ->
tokenize (loss only on answer). Pads batches to longest in-batch. Filters sequences > max_len."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import arc_core as ac
from augment import augment_task
from tokenizer import encode_train_example, PAD

class TaskStream:
    def __init__(self, split="training", max_len=2048, seed=0):
        self.tasks = list(ac.load_tasks(split).values())
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)

    def example(self):
        for _ in range(64):
            t = self.tasks[self.rng.integers(len(self.tasks))]
            aug, _ = augment_task(t, self.rng)
            tr = aug["train"]
            if len(tr) < 2:
                continue
            qi = int(self.rng.integers(len(tr)))
            demos = [tr[j] for j in range(len(tr)) if j != qi]
            enc = encode_train_example(demos, tr[qi]["input"], tr[qi]["output"], self.max_len)
            if enc is not None:
                return enc
        return None

    def batch(self, bs):
        items = []
        while len(items) < bs:
            e = self.example()
            if e is not None:
                items.append(e)
        T = max(len(ids) for ids, _ in items)
        ids = np.full((bs, T), PAD, dtype=np.int64)
        mask = np.zeros((bs, T), dtype=np.float32)
        for i, (x, m) in enumerate(items):
            ids[i, :len(x)] = x; mask[i, :len(m)] = m
        return ids, mask
