# Configurable CNN for the architecture search (Chinese Standard Mahjong).
# Superset of the caiest CNNModel: channels, depth, head type are configurable.
# A model built with cfg {channels:128, blocks:16, head:'flatten'} is the caiest baseline
# and is state-dict-compatible with model.py's CNNModel (same layer names), so any winning
# config in the {128,16,flatten} family can deploy via the existing bot/model.py unchanged.
import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.ReLU(),
        )
    def forward(self, x):
        return x + self._conv(x)

class CfgCNN(nn.Module):
    def __init__(self, channels=128, blocks=16, head='flatten', in_planes=38):
        super().__init__()
        self.head_kind = head
        self.head = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.ReLU(),
        )
        self.body = nn.Sequential(*(Bottleneck(channels) for _ in range(blocks)))
        if head == 'flatten':
            self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels*4*9, 512), nn.ReLU(), nn.Linear(512, 235))
        elif head == 'gap':  # global average pool -> smaller, more regularized
            self.foot = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 512), nn.ReLU(), nn.Linear(512, 235))
        else:
            raise ValueError(head)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        x = input_dict["obs"]["observation"].float()
        x = self.foot(self.body(self.head(x)))
        mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        return x + inf_mask
