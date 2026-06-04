# Minimal BN-free fused ResNet for Botzone deploy (Python 3.6 / torch 1.4 safe).
# Only Conv2d/Linear/ReLU/Flatten + clamp/log — no BatchNorm, no Transformer, no fancy ops.
# Numerically identical (eval) to the trained resbn40; weights = resbn40_fused.pkl.
import torch
from torch import nn

class _FusedBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
    def forward(self, x):
        y = torch.relu(self.c1(x)); y = self.c2(y); return torch.relu(x + y)

class ResFused(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__()
        self.stem = nn.Conv2d(38, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*(_FusedBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 9, 512), nn.ReLU(), nn.Linear(512, 235))
    def forward(self, input_dict):
        obs = input_dict['obs']['observation'].float()
        x = torch.relu(self.stem(obs))
        logits = self.foot(self.body(x))
        mask = input_dict['obs']['action_mask'].float()
        return logits + torch.clamp(torch.log(mask), -1e38, 1e38)
