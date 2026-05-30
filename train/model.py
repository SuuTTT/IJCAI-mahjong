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

class NumpyMLP:
    """
    Fast numpy forward pass. Load from .npz checkpoint.
    Used in the bot at serve time.
    """
    def __init__(self, path: str):
        data = np.load(path)
        self.layers = []
        i = 0
        while f"w{i}" in data:
            self.layers.append((data[f"w{i}"], data[f"b{i}"]))
            i += 1
        self.policy_w = data["policy_w"]
        self.policy_b = data["policy_b"]
        self.value_w  = data["value_w"]
        self.value_b  = data["value_b"]
        # Optional two-layer value head
        if "value_w0" in data:
            self.value_w0 = data["value_w0"]
            self.value_b0 = data["value_b0"]

    def forward(self, obs: np.ndarray, mask: np.ndarray = None):
        """
        obs:  (240,) uint8
        mask: (235,) bool, optional legal action mask
        Returns: (policy_probs (235,), value scalar)
        """
        x = obs.astype(np.float32) / 255.0

        for w, b in self.layers:
            x = np.maximum(0, x @ w.T + b)  # ReLU

        logits = x @ self.policy_w.T + self.policy_b  # (235,)
        # value head: hidden -> 64 (ReLU) -> 1 (Tanh)
        if hasattr(self, 'value_w0'):
            h = np.maximum(0, x @ self.value_w0.T + self.value_b0)
            value = float(np.tanh(h @ self.value_w.T + self.value_b))
        else:
            value = 0.0  # fallback if not exported

        if mask is not None:
            logits = np.where(mask, logits, -1e9)

        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        return probs, value

    def best_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        probs, _ = self.forward(obs, mask)
        return int(np.argmax(probs))

    @classmethod
    def random_init(cls, hidden=(512, 512, 256)):
        """Create and save a random-weight model for testing the pipeline."""
        import tempfile
        arrays = {}
        dims = [OBS_DIM] + list(hidden)
        for i, (in_d, out_d) in enumerate(zip(dims, dims[1:])):
            scale = np.sqrt(2.0 / in_d)
            arrays[f"w{i}"] = (np.random.randn(out_d, in_d) * scale).astype(np.float32)
            arrays[f"b{i}"] = np.zeros(out_d, dtype=np.float32)
        last = dims[-1]
        arrays["policy_w"] = (np.random.randn(ACT_DIM, last) * 0.01).astype(np.float32)
        arrays["policy_b"] = np.zeros(ACT_DIM, dtype=np.float32)
        arrays["value_w"]  = (np.random.randn(1, last) * 0.01).astype(np.float32)
        arrays["value_b"]  = np.zeros(1, dtype=np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez(tmp.name, **arrays)
        return cls(tmp.name), tmp.name


# ── PyTorch model (training only) ─────────────────────────────────────────────

if HAS_TORCH:
    class ResBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d), nn.LayerNorm(d), nn.ReLU(),
                nn.Linear(d, d), nn.LayerNorm(d),
            )

        def forward(self, x):
            return F.relu(x + self.net(x))


    class MahjongNet(nn.Module):
        """
        Policy + value head.
        obs -> stem -> residual blocks -> policy_head + value_head
        """
        def __init__(self, hidden=512, n_blocks=6):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(OBS_DIM, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )
            self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
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
            """Export weights to numpy .npz for serve-time inference."""
            arrays = {}
            # stem + blocks flatten to linear layers
            param_layers = []
            # stem
            param_layers.append(
                (self.stem[0].weight.detach().cpu().numpy(),
                 self.stem[0].bias.detach().cpu().numpy())
            )
            # blocks
            for block in self.blocks:
                for lin in [block.net[0], block.net[3]]:
                    param_layers.append(
                        (lin.weight.detach().cpu().numpy(),
                         lin.bias.detach().cpu().numpy())
                    )

            for i, (w, b) in enumerate(param_layers):
                arrays[f"w{i}"] = w.astype(np.float32)
                arrays[f"b{i}"] = b.astype(np.float32)

            arrays["policy_w"] = self.policy_head.weight.detach().cpu().numpy().astype(np.float32)
            arrays["policy_b"] = self.policy_head.bias.detach().cpu().numpy().astype(np.float32)
            # value_head: Linear(hidden,64) -> ReLU -> Linear(64,1) -> Tanh
            arrays["value_w0"] = self.value_head[0].weight.detach().cpu().numpy().astype(np.float32)
            arrays["value_b0"] = self.value_head[0].bias.detach().cpu().numpy().astype(np.float32)
            arrays["value_w"]  = self.value_head[2].weight.detach().cpu().numpy().astype(np.float32)
            arrays["value_b"]  = self.value_head[2].bias.detach().cpu().numpy().astype(np.float32)

            np.savez(path, **arrays)
            print(f"Exported numpy weights to {path}")
