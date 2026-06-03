# Policy+Value CNN for RL fine-tuning the Mahjong SL base.
# Shares the SL CNN backbone (stem + 16 residual blocks), adds a value head alongside the
# existing policy head. load_sl() copies the converged SL weights into the shared backbone +
# policy head, leaving the value head fresh — so RL starts from the strong supervised policy.
import torch
from torch import nn

CONV = 128
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(CONV, CONV, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(CONV, CONV, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(CONV, CONV, 3, 1, 1, bias=False), nn.ReLU())
    def forward(self, x): return x + self._conv(x)

class CNNPolicyValue(nn.Module):
    """Same backbone/naming as model.py CNNModel (head/body/foot) for state_dict compat,
    plus a value head (vfoot). forward -> (masked_logits, value)."""
    def __init__(self, blocks=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(38, CONV, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(CONV, CONV, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(CONV, CONV, 3, 1, 1, bias=False), nn.ReLU())
        self.body = nn.Sequential(*(Bottleneck() for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(CONV * 4 * 9, 512), nn.ReLU(), nn.Linear(512, 235))
        self.vfoot = nn.Sequential(nn.Flatten(), nn.Linear(CONV * 4 * 9, 256), nn.ReLU(), nn.Linear(256, 1))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(m.weight)

    def features(self, obs):
        return self.body(self.head(obs.float()))

    def forward(self, obs, action_mask):
        f = self.features(obs)
        logits = self.foot(f)
        inf_mask = torch.clamp(torch.log(action_mask.float()), -1e38, 1e38)
        v = self.vfoot(f).squeeze(-1)
        return logits + inf_mask, v

    def load_sl(self, sl_state):
        """Copy SL CNNModel weights (head/body/foot) into this net; vfoot stays fresh."""
        own = self.state_dict()
        copied = 0
        for k, val in sl_state.items():
            if k in own and own[k].shape == val.shape:
                own[k] = val; copied += 1
        self.load_state_dict(own)
        return copied

if __name__ == '__main__':
    import sys, os
    m = CNNPolicyValue()
    sl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'arch_ck', 'base_16x128_final.pkl')
    n = m.load_sl(torch.load(sl, map_location='cpu'))
    x = torch.zeros(2, 38, 4, 9); mk = torch.ones(2, 235)
    lg, v = m(x, mk)
    print(f"loaded {n} SL tensors; forward logits {tuple(lg.shape)} value {tuple(v.shape)} params {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
