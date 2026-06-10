"""Tiny decoder-only transformer (GPT-style) for ARC sequence modeling. ~14M params @ d=384,h=6,L=8.
Uses F.scaled_dot_product_attention (flash on Blackwell). bf16-friendly. Knowledge-free: trained only on
ARC-train + augmentation, no external pretraining."""
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizer import VOCAB

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = h

    def forward(self, x):
        B, T, D = x.shape
        y = self.ln1(x)
        qkv = self.qkv(y).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab=VOCAB, d=384, h=6, L=8, max_len=2048):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.max_len = max_len
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
