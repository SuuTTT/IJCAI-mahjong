# Exploration architectures for the Mahjong SL search (input (B,38,4,9) -> 235 logits).
# All share the caiest feature (38 planes, 4x9 tile grid) and the masked-logits output, so any
# winner is a drop-in for the deploy bot's forward (only model.py's class body changes).
#
#   build(kind, **cfg) -> nn.Module with .forward(input_dict) like CfgCNN.
# kinds: 'resbn' (BatchNorm CNN, enables depth), 'attn' (tile-token transformer),
#        'cnn_attn' (CNN stem + transformer), 'gnn' (tile-graph GCN over 34 tile-types).
import math
import torch
from torch import nn

IN_PLANES = 38
GRID = 4 * 9  # 36 tile positions

def _mask(logits, action_mask):
    inf_mask = torch.clamp(torch.log(action_mask.float()), -1e38, 1e38)
    return logits + inf_mask

# ---- 1) Residual CNN with BatchNorm (lets us go deep without divergence) ----
class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)

class ResBNCNN(nn.Module):
    def __init__(self, channels=128, blocks=24, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU(), nn.Linear(512, 235))
    def forward(self, d):
        self.train(d.get('is_training', False))
        x = d['obs']['observation'].float()
        return _mask(self.foot(self.body(self.stem(x))), d['obs']['action_mask'])

# ---- 2) Tile-token Transformer (36 positions as tokens, 38-dim features) ----
class TileTransformer(nn.Module):
    def __init__(self, d_model=128, layers=6, heads=8, **_):
        super().__init__()
        self.proj = nn.Linear(IN_PLANES, d_model)
        self.pos = nn.Parameter(torch.zeros(1, GRID, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout=0.1,
                                         batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 512), nn.ReLU(), nn.Linear(512, 235))
        nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, d):
        self.train(d.get('is_training', False))
        x = d['obs']['observation'].float()           # (B,38,4,9)
        B = x.shape[0]
        x = x.view(B, IN_PLANES, GRID).transpose(1, 2)  # (B,36,38)
        x = self.proj(x) + self.pos
        x = self.enc(x).mean(dim=1)                     # pool over tokens
        return _mask(self.head(x), d['obs']['action_mask'])

# ---- 3) CNN stem + Transformer (local conv features, then global attention) ----
class CNNTransformer(nn.Module):
    def __init__(self, channels=128, conv_blocks=4, layers=4, heads=8, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.conv = nn.Sequential(*(_BNBlock(channels) for _ in range(conv_blocks)))
        self.pos = nn.Parameter(torch.zeros(1, GRID, channels)); nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(channels, heads, channels * 4, dropout=0.1, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 512), nn.ReLU(), nn.Linear(512, 235))
    def forward(self, d):
        self.train(d.get('is_training', False))
        x = d['obs']['observation'].float()
        B = x.shape[0]
        x = self.conv(self.stem(x))                     # (B,C,4,9)
        x = x.view(B, x.shape[1], GRID).transpose(1, 2) + self.pos
        x = self.enc(x).mean(dim=1)
        return _mask(self.head(x), d['obs']['action_mask'])

# ---- 4) Tile-type GNN: 34 tile-type nodes, edges = suit-adjacency + same-type ----
class TileGNN(nn.Module):
    """Aggregate the 38 planes per tile-type into node features, message-pass over a fixed
    tile graph (sequence neighbours within a suit + honor cliques), then pool to 235."""
    def __init__(self, hidden=256, layers=4, **_):
        super().__init__()
        self.A = nn.Parameter(self._adj(), requires_grad=False)   # (34,34) normalized adjacency
        self.inp = nn.Linear(IN_PLANES, hidden)
        self.gc = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden * 34, 512), nn.ReLU(), nn.Linear(512, 235))
    @staticmethod
    def _adj():
        import numpy as np
        A = np.eye(34)
        # suits W(0-8) T(9-17) B(18-26): connect n with n+-1, n+-2 within suit
        for s in range(3):
            base = s * 9
            for i in range(9):
                for d in (1, 2):
                    if i + d < 9: A[base + i, base + i + d] = A[base + i + d, base + i] = 1
        # honors 27-33 (winds F1-4, dragons J1-3): clique
        for i in range(27, 34):
            for j in range(27, 34): A[i, j] = 1
        D = A.sum(1, keepdims=True) ** -0.5
        return torch.tensor((A * D * D.T), dtype=torch.float32)
    def forward(self, d):
        self.train(d.get('is_training', False))
        x = d['obs']['observation'].float()             # (B,38,4,9)
        B = x.shape[0]
        # map 36 grid positions -> 34 tile types (first 34 of the 36)
        x = x.view(B, IN_PLANES, GRID)[:, :, :34].transpose(1, 2)   # (B,34,38)
        h = torch.relu(self.inp(x))
        for gc in self.gc:
            h = torch.relu(gc(self.A @ h))               # A:(34,34) @ h:(B,34,hid) broadcast
        return _mask(self.head(h), d['obs']['action_mask'])

# ---- BN-free fused version of ResBNCNN (Conv+BN folded -> Conv with bias) ----
# Deployable on Botzone torch 1.4 (no BatchNorm running-stat / version issues). In eval mode
# the fused model is numerically identical to the trained ResBNCNN.
class _FusedBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
    def forward(self, x):
        y = torch.relu(self.c1(x)); y = self.c2(y); return torch.relu(x + y)

class ResFused(nn.Module):
    def __init__(self, channels=128, blocks=40, **_):
        super().__init__()
        self.stem = nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*(_FusedBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU(), nn.Linear(512, 235))
    def forward(self, d):
        self.train(d.get('is_training', False))
        x = torch.relu(self.stem(d['obs']['observation'].float()))
        return _mask(self.foot(self.body(x)), d['obs']['action_mask'])

def fuse_resbn(resbn):
    """Return a ResFused with the same channels/blocks, weights = fuse(Conv,BN) of `resbn` (eval)."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    resbn.eval()
    ch = resbn.stem[0].out_channels; blocks = len(resbn.body)
    f = ResFused(channels=ch, blocks=blocks).eval()
    fs = fuse_conv_bn_eval(resbn.stem[0], resbn.stem[1])
    f.stem.load_state_dict(fs.state_dict())
    for i, blk in enumerate(resbn.body):
        c1 = fuse_conv_bn_eval(blk.c1, blk.b1); c2 = fuse_conv_bn_eval(blk.c2, blk.b2)
        f.body[i].c1.load_state_dict(c1.state_dict()); f.body[i].c2.load_state_dict(c2.state_dict())
    f.foot.load_state_dict(resbn.foot.state_dict())
    return f

def build(kind, **cfg):
    return {'resbn': ResBNCNN, 'resbn_fused': ResFused, 'attn': TileTransformer,
            'cnn_attn': CNNTransformer, 'gnn': TileGNN}[kind](**cfg)
