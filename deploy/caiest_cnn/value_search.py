# value_search.py — 1-ply value-guided discard selection (torch-1.4-safe, opt-in CAIEST_VNET).
# For each near-top legal discard, build the RESULTING hand-state obs (remove that tile from the
# HAND thermometer planes 2..5) and score it with V (final-duplicate-score regressor, r~0.72,
# trained on real top-30 outcomes — so it implicitly weighs deal-in risk + fan value). Pick the
# discard with the best resulting V. Unlike Q-rerank (which scored a confounded action embedding),
# this scores the actual future STATE. Claims/Hu/Pass never touched; stays among policy top-k to
# remain in-distribution. Cost = k value-forwards/turn (~3-6) — fits Botzone's 6s budget.
import os
import numpy as np
import torch
from torch import nn

HAND0 = 2          # OFFSET_OBS['HAND'] — hand thermometer occupies planes 2,3,4,5


class _FusedBlock(nn.Module):
    def __init__(self, ch):
        super(_FusedBlock, self).__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)

    def forward(self, x):
        y = torch.relu(self.c1(x)); y = self.c2(y); return torch.relu(x + y)


class ValueNet(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super(ValueNet, self).__init__()
        self.stem = nn.Conv2d(38, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[_FusedBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def v(self, obs_batch):                 # obs_batch (B,38,4,9) float
        x = torch.relu(self.stem(obs_batch))
        x = self.body(x).mean(3).mean(2)    # GAP (torch-1.4-safe)
        return self.head(x).squeeze(1)


def load(path, blocks=40):
    net = ValueNet(blocks=blocks); net.load_state_dict(torch.load(path, map_location='cpu')); net.eval()
    return net


K = int(os.environ.get('CAIEST_VK', '4'))
DELTA = float(os.environ.get('CAIEST_VDELTA', '2.0'))


def _drop_tile(obs, t):
    """Return obs with one copy of tile index `t` (0..33) removed from the HAND thermometer.
    obs is (38,4,9); tile t sits at cell (t//9, t%9). Clear the highest set hand-plane there."""
    o = obs.copy(); r, c = t // 9, t % 9
    for p in range(HAND0 + 3, HAND0 - 1, -1):      # planes 5..2, clear the highest set one
        if o[p, r, c]:
            o[p, r, c] = 0
            break
    return o


def pick_discard(vnet, obs_np, mask_np, logits_np, play_offset, tile_of):
    """obs_np (38,4,9); tile_of(action_idx)->tile col index 0..33. Return reranked Play idx or None."""
    lg = np.asarray(logits_np).flatten()
    legal = [i for i in range(play_offset, play_offset + 34) if mask_np[i]]
    if len(legal) < 2:
        return None
    legal.sort(key=lambda i: -float(lg[i]))
    cands = [i for i in legal[:K] if lg[legal[0]] - lg[i] <= DELTA]
    if len(cands) < 2:
        return None
    batch = np.stack([_drop_tile(obs_np, tile_of(i)) for i in cands]).astype(np.float32)
    with torch.no_grad():
        v = vnet.v(torch.from_numpy(batch))
    return cands[int(v.argmax())]
