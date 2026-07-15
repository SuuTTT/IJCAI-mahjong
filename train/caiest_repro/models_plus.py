"""
models_plus.py — ResBNCNN with a configurable number of input planes (38 + extra).
Identical to models_explore.ResBNCNN except IN_PLANES is a constructor arg, so a candidate
trained on augmented features (38+P planes) loads/builds correctly. Base-38 (moyu) still uses
models_explore.build('resbn', ...).
"""
import torch
from torch import nn

GRID = 4 * 9

def _mask(logits, action_mask):
    inf_mask = torch.clamp(torch.log(action_mask.float()), -1e38, 1e38)
    return logits + inf_mask

class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)

class ResBNCNNP(nn.Module):
    def __init__(self, in_planes=38, channels=128, blocks=40, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU(), nn.Linear(512, 235))
    def forward(self, d):
        self.train(d.get('is_training', False))
        x = d['obs']['observation'].float()
        return _mask(self.foot(self.body(self.stem(x))), d['obs']['action_mask'])

def build_plus(in_planes=38, channels=128, blocks=40):
    return ResBNCNNP(in_planes=in_planes, channels=channels, blocks=blocks)
