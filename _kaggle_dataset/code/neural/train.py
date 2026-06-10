"""Train the tiny ARC transformer on the 5080 (bf16). Loss only on answer tokens. Saves a checkpoint."""
import sys, os, time, math, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, torch.nn.functional as F
from model import GPT
from dataset import TaskStream
from tokenizer import VOCAB

def keep_awake():
    """Prevent Windows system sleep while training (CUDA contexts die on sleep). Scoped to this process."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("keep-awake: system sleep inhibited while training", flush=True)
    except Exception as e:
        print(f"keep-awake unavailable: {e}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=3000)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ckpt", type=str, default="experiments/gpt.pt")
    a = ap.parse_args()

    keep_awake()
    dev = "cuda"
    torch.manual_seed(0)
    stream = TaskStream("training", max_len=a.max_len)
    model = GPT(VOCAB, a.d, a.heads, a.layers, a.max_len).to(dev)
    if a.resume and os.path.exists(a.resume):
        model.load_state_dict(torch.load(a.resume, map_location=dev)["sd"])
        print(f"resumed weights from {a.resume}", flush=True)
    print(f"params {model.n_params()/1e6:.1f}M | bs {a.bs} max_len {a.max_len} steps {a.steps}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(s):
        if s < a.warmup: return a.lr * s / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 0.1 * a.lr + 0.9 * a.lr * 0.5 * (1 + math.cos(math.pi * p))

    t0 = time.time(); run_loss = run_acc = 0.0
    for step in range(1, a.steps + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        ids, mask = stream.batch(a.bs)
        ids = torch.from_numpy(ids).to(dev); mask = torch.from_numpy(mask).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids[:, :-1])
            tgt = ids[:, 1:]; m = mask[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
            loss = (ce * m.reshape(-1)).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = (((pred == tgt) * m).sum() / m.sum().clamp(min=1)).item()
        run_loss += loss.item(); run_acc += acc
        if step % 50 == 0 or step == 1:
            n = 50 if step % 50 == 0 else 1
            print(f"step {step:5d} | loss {run_loss/n:.4f} | tok_acc {run_acc/n:.3f} | "
                  f"{step/(time.time()-t0):.2f} it/s | mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            run_loss = run_acc = 0.0
        if step % a.save_every == 0:
            os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
            torch.save({"sd": model.state_dict(), "cfg": vars(a)}, a.ckpt)
            print(f"  [ckpt saved @ step {step}]", flush=True)
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
    torch.save({"sd": model.state_dict(), "cfg": vars(a)}, a.ckpt)
    print(f"saved {a.ckpt} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
