# Q-rerank (test-time search, minimal form) — torch-1.4-safe, OPT-IN via CAIEST_QNET.
# Among the policy's near-top legal DISCARDS (logit within CAIEST_QDELTA of the best discard,
# capped at CAIEST_QK), pick the one with the highest learned Q(s,a) = expected final duplicate
# score (q_head.py, trained on top-30 seats' real outcomes). Claims/Hu/Pass are NEVER touched.
# Staying near-policy keeps Q on-distribution (it is an on-policy outcome model, not a full Q*).
import os
import torch
from torch import nn


class _FusedBlock(nn.Module):
    def __init__(self, ch):
        super(_FusedBlock, self).__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)

    def forward(self, x):
        y = torch.relu(self.c1(x))
        y = self.c2(y)
        return torch.relu(x + y)


class QNet(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super(QNet, self).__init__()
        self.stem = nn.Conv2d(38, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[_FusedBlock(channels) for _ in range(blocks)])
        self.emb = nn.Embedding(235, 64)
        self.head = nn.Sequential(nn.Linear(channels + 64, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def q_for(self, obs, acts):
        """obs (1,38,4,9) float tensor, acts: LongTensor (K,) -> (K,) Q values. One trunk pass."""
        x = torch.relu(self.stem(obs))
        f = self.body(x).mean(3).mean(2)               # (1,ch) — torch-1.4-safe GAP
        f = f.expand(acts.size(0), f.size(1))
        return self.head(torch.cat([f, self.emb(acts)], 1)).squeeze(1)


def load(path, blocks=40):
    net = QNet(blocks=blocks)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()
    return net


K = int(os.environ.get('CAIEST_QK', '4'))
DELTA = float(os.environ.get('CAIEST_QDELTA', '1.5'))


def pick_discard(qnet, obs_np, mask_np, logits_np, play_offset):
    """Return the reranked Play action index, or None to keep the policy's choice."""
    import numpy as np
    lg = np.asarray(logits_np).flatten()
    legal_play = [i for i in range(play_offset, play_offset + 34) if mask_np[i]]
    if len(legal_play) < 2:
        return None
    legal_play.sort(key=lambda i: -float(lg[i]))
    best = legal_play[0]
    cands = [i for i in legal_play[:K] if lg[best] - lg[i] <= DELTA]
    if len(cands) < 2:
        return None
    with torch.no_grad():
        ob = torch.from_numpy(np.expand_dims(obs_np, 0)).float()
        q = qnet.q_for(ob, torch.LongTensor(cands))
    return cands[int(q.argmax())]
