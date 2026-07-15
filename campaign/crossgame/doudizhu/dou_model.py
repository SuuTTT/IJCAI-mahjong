"""Shared Doudizhu student net + legal-mask helper. obs(790 float) -> 27472 logits."""
import torch
import torch.nn as nn

N_ACT = 27472
OBS_DIM = 901  # doudizhu state_shape [[790],[901],[901]]; 790 landlord obs zero-padded to 901

class DouMLP(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_act=N_ACT, hidden=1024, layers=3):
        super().__init__()
        blocks = []
        d = obs_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, n_act)

    def forward(self, x):
        return self.head(self.trunk(x))

def make_mask(legal, n_act=N_ACT):
    """legal (B,MAXLEGAL) int with -1 pad -> bool mask (B,n_act).
    Pad ids are routed to a dummy column then sliced off, so a padded 0 never
    accidentally marks action 0 legal."""
    idx = torch.where(legal >= 0, legal, torch.full_like(legal, n_act)).long()
    mask = torch.zeros(legal.shape[0], n_act + 1, dtype=torch.bool, device=legal.device)
    mask.scatter_(1, idx, True)
    return mask[:, :n_act]

def masked_logits(logits, mask):
    return torch.where(mask, logits.float(), torch.full_like(logits, -1e30, dtype=torch.float32))
