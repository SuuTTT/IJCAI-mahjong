"""Small CNN policy net for 6x6 Othello + board->planes encoding."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NSQ = 36
_SQS = np.arange(NSQ, dtype=np.int64)


def encode_planes(me_arr, opp_arr):
    """me_arr, opp_arr: int64 arrays of packed bitboards (mover perspective).
    Returns float32 array [n,3,6,6]: planes own, opp, empty."""
    me_arr = np.asarray(me_arr, dtype=np.int64)
    opp_arr = np.asarray(opp_arr, dtype=np.int64)
    me_bits = ((me_arr[:, None] >> _SQS[None, :]) & 1).astype(np.float32)
    opp_bits = ((opp_arr[:, None] >> _SQS[None, :]) & 1).astype(np.float32)
    empty = 1.0 - me_bits - opp_bits
    X = np.stack([me_bits, opp_bits, empty], axis=1).reshape(-1, 3, 6, 6)
    return X


def legal_mask_from_ints(me, opp, oth):
    """Return a bool length-36 array of legal squares for a single (me,opp)."""
    m = oth.legal_moves(me, opp)
    mask = np.zeros(NSQ, dtype=bool)
    for s in range(NSQ):
        if (m >> s) & 1:
            mask[s] = True
    return mask


class PolicyCNN(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b3 = nn.BatchNorm2d(ch)
        self.head = nn.Conv2d(ch, 1, 1)   # -> [n,1,6,6] -> 36 logits
        self.bias = nn.Parameter(torch.zeros(NSQ))

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = self.head(x).flatten(1)       # [n,36]
        return x + self.bias
