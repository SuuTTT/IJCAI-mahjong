"""Torch policy + critic modules for the T2 PPO fine-tune of the MCR Mahjong SL policy.

PolicyNet  -- BN-free fused ResNet, architecture identical to mcr_champion/model_resfused.py
              (``ResFused``).  Loads ``/root/mcr_t2/kd_128x40_s0.pkl`` (the SL student) EXACTLY:
              keys ``stem.{weight,bias}``, ``body.{i}.c{1,2}.{weight,bias}``, ``foot.{1,3}.*``.
              forward(x) -> raw action logits (235,); the trainer applies the legal mask.

ValueNet   -- ResBN-128x40 trunk WITH BatchNorm (keys ``stem.0``=conv, ``stem.1``=BN,
              ``body.{i}.c{1,2}`` convs, ``body.{i}.b{1,2}`` BNs) + value head
              ``foot.1 (512,4608)`` -> ``foot.3 (1,512)``.  Loads
              ``/root/mcr_t2/value_e2e_ckpt.pt`` EXACTLY.  It predicts score/30
              (``score_scale=30``); the trainer rescales to the reward scale (score/8).

Both are reconstructed from the state_dict shapes verified on the box.
"""
import torch
from torch import nn

CH = 128
FLAT = CH * 4 * 9          # 4608
N_ACT = 235
N_IN = 38


# ------------------------------------------------------------------ policy (BN-free)
class _FusedBlock(nn.Module):
    def __init__(self, ch=CH):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)

    def forward(self, x):
        y = torch.relu(self.c1(x))
        y = self.c2(y)
        return torch.relu(x + y)


class PolicyNet(nn.Module):
    """BN-free fused ResNet.  forward -> RAW logits (no mask applied)."""

    def __init__(self, channels=CH, blocks=40):
        super().__init__()
        self.stem = nn.Conv2d(N_IN, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*(_FusedBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, N_ACT),
        )

    def forward(self, obs):
        x = torch.relu(self.stem(obs))
        return self.foot(self.body(x))


# ------------------------------------------------------------------ critic (with BN)
class _BNBlock(nn.Module):
    def __init__(self, ch=CH):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class ValueNet(nn.Module):
    """ResBN trunk + scalar value head.  Raw output is the score/30 prediction."""

    def __init__(self, channels=CH, blocks=40):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_IN, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, obs):
        return self.foot(self.body(self.stem(obs))).squeeze(-1)


def load_policy(path, device="cpu", channels=CH, blocks=40):
    sd = torch.load(path, map_location="cpu")
    m = PolicyNet(channels, blocks)
    missing, unexpected = m.load_state_dict(sd, strict=True)
    m.to(device)
    return m


def load_value(path, device="cpu", channels=CH, blocks=40):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and "stem.0.weight" not in sd:
        sd = sd["state_dict"]
    m = ValueNet(channels, blocks)
    m.load_state_dict(sd, strict=True)
    m.to(device)
    return m
