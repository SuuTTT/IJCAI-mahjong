"""
model.py — MLP policy + value network for mahjong.

Input:  240-dim uint8 observation (normalized to float32)
Output: policy logits (235) + value scalar (1)

Designed to run without GPU at inference time:
  weights exported as numpy .npz, forward pass in pure numpy (~1ms CPU).
"""

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

OBS_DIM = 240
ACT_DIM = 235
EMPTY   = 255  # sentinel value in obs


# ── Numpy-only inference (no PyTorch required at serve time) ──────────────────

def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps=1e-5) -> np.ndarray:
    mu  = x.mean()
    std = np.sqrt(x.var() + eps)
    return (x - mu) / std * g + b


class NumpyMLP:
    """
    Faithful numpy forward pass matching MahjongNet PyTorch architecture.

    Stored format (npz):
      stem_w, stem_b              — stem Linear weights
      stem_ln_g, stem_ln_b        — stem LayerNorm
      block{i}_w1, block{i}_b1   — ResBlock Linear1
      block{i}_ln1_g/b            — ResBlock LayerNorm1
      block{i}_w2, block{i}_b2   — ResBlock Linear2
      block{i}_ln2_g/b            — ResBlock LayerNorm2
      policy_w, policy_b          — policy head
      value_w0/b0, value_w/b      — value head two-layer
    """

    def __init__(self, path: str):
        d = np.load(path)
        self._d = d
        # stem
        self.stem_w  = d["stem_w"];  self.stem_b  = d["stem_b"]
        self.stem_g  = d["stem_ln_g"]; self.stem_bg = d["stem_ln_b"]
        # blocks
        self.blocks = []
        i = 0
        while f"block{i}_w1" in d:
            self.blocks.append({
                "w1": d[f"block{i}_w1"], "b1": d[f"block{i}_b1"],
                "g1": d[f"block{i}_ln1_g"], "bg1": d[f"block{i}_ln1_b"],
                "w2": d[f"block{i}_w2"], "b2": d[f"block{i}_b2"],
                "g2": d[f"block{i}_ln2_g"], "bg2": d[f"block{i}_ln2_b"],
            })
            i += 1
        # heads
        self.policy_w = d["policy_w"]; self.policy_b = d["policy_b"]
        self.value_w0 = d.get("value_w0"); self.value_b0 = d.get("value_b0")
        self.value_w  = d["value_w"];  self.value_b  = d["value_b"]

    def forward(self, obs: np.ndarray, mask: np.ndarray = None):
        """obs: (240,) uint8 → (probs (235,), value scalar)"""
        x = obs.astype(np.float32) / 255.0
        # Stem: Linear → LN → ReLU
        x = np.maximum(0, _layer_norm(x @ self.stem_w.T + self.stem_b,
                                      self.stem_g, self.stem_bg))
        # ResBlocks: res = ReLU(x + LN2(Linear2(ReLU(LN1(Linear1(x))))))
        for blk in self.blocks:
            h = _layer_norm(x @ blk["w1"].T + blk["b1"], blk["g1"], blk["bg1"])
            h = np.maximum(0, h)
            h = _layer_norm(h @ blk["w2"].T + blk["b2"], blk["g2"], blk["bg2"])
            x = np.maximum(0, x + h)
        # Policy head
        logits = x @ self.policy_w.T + self.policy_b
        if mask is not None:
            logits = np.where(mask, logits, -1e9)
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        # Value head: Linear → ReLU → Linear → Tanh
        if self.value_w0 is not None:
            h = np.maximum(0, x @ self.value_w0.T + self.value_b0)
            value = float(np.tanh(h @ self.value_w.T + self.value_b))
        else:
            value = 0.0
        return probs, value

    def best_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        probs, _ = self.forward(obs, mask)
        return int(np.argmax(probs))


# ── PyTorch model (training only) ─────────────────────────────────────────────

if HAS_TORCH:
    class ResBlock(nn.Module):
        def __init__(self, d, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d, d), nn.LayerNorm(d),
            )

        def forward(self, x):
            return F.relu(x + self.net(x))


    class MahjongNet(nn.Module):
        """
        Policy + value head with dropout regularisation.
        obs -> stem -> residual blocks -> policy_head + value_head
        """
        def __init__(self, hidden=512, n_blocks=6, dropout=0.1):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(OBS_DIM, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
            self.policy_head = nn.Linear(hidden, ACT_DIM)
            self.value_head  = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                             nn.Linear(64, 1), nn.Tanh())

        def forward(self, obs, mask=None):
            """
            obs:  (B, 240) float32
            mask: (B, 235) bool, optional
            Returns: (logits (B,235), value (B,1))
            """
            x = self.stem(obs / 255.0)
            x = self.blocks(x)
            logits = self.policy_head(x)
            if mask is not None:
                logits = logits.masked_fill(~mask, float("-inf"))
            value = self.value_head(x)
            return logits, value

        def export_numpy(self, path: str):
            """Export weights to numpy .npz matching NumpyMLP format."""
            def t(x): return x.detach().cpu().numpy().astype(np.float32)
            a = {}
            # Stem
            a["stem_w"]    = t(self.stem[0].weight)
            a["stem_b"]    = t(self.stem[0].bias)
            a["stem_ln_g"] = t(self.stem[1].weight)  # LayerNorm gamma
            a["stem_ln_b"] = t(self.stem[1].bias)    # LayerNorm beta
            # ResBlocks — net: [0=Lin1, 1=LN1, 2=ReLU, 3=Lin2 (or 4 with dropout), 4/5=LN2]
            for bi, block in enumerate(self.blocks):
                net = block.net
                # Find linears and layernorms by type
                lins = [m for m in net if isinstance(m, nn.Linear)]
                lns  = [m for m in net if isinstance(m, nn.LayerNorm)]
                a[f"block{bi}_w1"]    = t(lins[0].weight)
                a[f"block{bi}_b1"]    = t(lins[0].bias)
                a[f"block{bi}_ln1_g"] = t(lns[0].weight)
                a[f"block{bi}_ln1_b"] = t(lns[0].bias)
                a[f"block{bi}_w2"]    = t(lins[1].weight)
                a[f"block{bi}_b2"]    = t(lins[1].bias)
                a[f"block{bi}_ln2_g"] = t(lns[1].weight)
                a[f"block{bi}_ln2_b"] = t(lns[1].bias)
            # Policy & value heads
            a["policy_w"]  = t(self.policy_head.weight)
            a["policy_b"]  = t(self.policy_head.bias)
            a["value_w0"]  = t(self.value_head[0].weight)
            a["value_b0"]  = t(self.value_head[0].bias)
            a["value_w"]   = t(self.value_head[2].weight)
            a["value_b"]   = t(self.value_head[2].bias)
            np.savez(path, **a)
            print(f"Exported numpy weights ({len(a)} arrays) -> {path}")
