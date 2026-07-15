"""Small MLP policy net over Leduc infoset features + feature encoding.

Feature vector (27 dims), injective over infosets:
  private one-hot (3) | public one-hot {none,J,Q,K} (4) | round one-hot (2)
  | round-0 history: 3 slots x {pad,call,raise} (9)
  | round-1 history: 3 slots x {pad,call,raise} (9)
"""
import numpy as np
import torch
import torch.nn as nn

FEAT_DIM = 27
NACT = 3
HIST_SLOTS = 3


def encode_key(key):
    """key = (priv, pub, round, r0hist, r1hist) -> float32[27]."""
    priv, pub, rnd, r0, r1 = key
    v = np.zeros(FEAT_DIM, dtype=np.float32)
    v[priv] = 1.0                          # 0..2
    v[3 + (0 if pub == -1 else pub + 1)] = 1.0   # 3..6
    v[7 + rnd] = 1.0                       # 7..8
    base = 9
    for hist, off in ((r0, base), (r1, base + 9)):
        for i in range(HIST_SLOTS):
            if i < len(hist):
                a = hist[i]                # 1=call, 2=raise
                slot = 1 if a == 1 else 2  # call->1, raise->2
            else:
                slot = 0                   # pad
            v[off + i * 3 + slot] = 1.0
    return v


def encode_keys(keys):
    return np.stack([encode_key(k) for k in keys], axis=0)


class PolicyMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, NACT),
        )

    def forward(self, x):
        return self.net(x)
