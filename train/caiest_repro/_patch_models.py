import re
P="models_explore.py"
s=open(P).read()
if "ResBNGNN" in s:
    print("already patched"); raise SystemExit
BLOCK=r"""

# ---- HYBRID: CNN backbone + PARALLEL tile-GNN branch (concatenated before the head) ----
# CNN branch is byte-identical to ResBNCNN (stem+body+first foot Linear->512): worst case the
# head learns to ignore the GNN and this TIES aug_s0. GNN branch reads per-tile features from the
# SAME (38,4,9) obs, so the on-GPU suit/reflect/dragon augmentation flows into BOTH branches
# consistently (all three augs are automorphisms of the tile graph). 3 message-passing layers over
# 34 tile-type nodes; edges = within-suit chi adjacency (r+-1,r+-2) + honor clique + self-loop
# (peng / same-tile); mean-pool -> emb; concat with the CNN 512-vec -> 235 head.
class _GNNBranch(nn.Module):
    def __init__(self, hidden=128, layers=3, emb=128):
        super().__init__()
        self.register_buffer("A", TileGNN._adj())          # (34,34) sym-normalized adjacency
        self.inp = nn.Linear(IN_PLANES, hidden)
        self.gc = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, emb)
    def forward(self, x):                                    # x:(B,38,4,9)
        B = x.shape[0]
        nf = x.view(B, IN_PLANES, GRID)[:, :, :34].transpose(1, 2)   # (B,34,38) per-tile feats
        h = torch.relu(self.inp(nf))
        for gc in self.gc:
            h = torch.relu(gc(self.A @ h))                   # message passing
        return torch.relu(self.out(h.mean(dim=1)))           # (B,emb) pooled graph embedding

class ResBNGNN(nn.Module):
    def __init__(self, channels=128, blocks=40, in_planes=IN_PLANES,
                 gnn_hidden=128, gnn_layers=3, gnn_emb=128, **_):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.cnn_fc = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU())
        self.gnn = _GNNBranch(gnn_hidden, gnn_layers, gnn_emb)
        self.head = nn.Linear(512 + gnn_emb, 235)
    def forward(self, d):
        self.train(d.get("is_training", False))
        x = d["obs"]["observation"].float()
        f = self.body(self.stem(x))
        cnn_vec = self.cnn_fc(f)
        g = self.gnn(x)
        return _mask(self.head(torch.cat([cnn_vec, g], 1)), d["obs"]["action_mask"])

class ResFusedGNN(nn.Module):
    """BN-folded ResBNGNN (deployable / gating). Numerically identical to ResBNGNN in eval."""
    def __init__(self, channels=128, blocks=40, in_planes=IN_PLANES,
                 gnn_hidden=128, gnn_layers=3, gnn_emb=128, **_):
        super().__init__()
        self.stem = nn.Conv2d(in_planes, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*(_FusedBlock(channels) for _ in range(blocks)))
        self.cnn_fc = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU())
        self.gnn = _GNNBranch(gnn_hidden, gnn_layers, gnn_emb)
        self.head = nn.Linear(512 + gnn_emb, 235)
    def forward(self, d):
        self.train(d.get("is_training", False))
        x = d["obs"]["observation"].float()
        f = self.body(torch.relu(self.stem(x)))
        cnn_vec = self.cnn_fc(f)
        g = self.gnn(x)
        return _mask(self.head(torch.cat([cnn_vec, g], 1)), d["obs"]["action_mask"])

def fuse_resbngnn(m):
    """Fold BN of a trained ResBNGNN -> ResFusedGNN (eval, torch-1.4-safe)."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    m.eval()
    ch = m.stem[0].out_channels; blocks = len(m.body); in_planes = m.stem[0].in_channels
    f = ResFusedGNN(channels=ch, blocks=blocks, in_planes=in_planes).eval()
    fs = fuse_conv_bn_eval(m.stem[0], m.stem[1]); f.stem.load_state_dict(fs.state_dict())
    for i, blk in enumerate(m.body):
        c1 = fuse_conv_bn_eval(blk.c1, blk.b1); c2 = fuse_conv_bn_eval(blk.c2, blk.b2)
        f.body[i].c1.load_state_dict(c1.state_dict()); f.body[i].c2.load_state_dict(c2.state_dict())
    f.cnn_fc.load_state_dict(m.cnn_fc.state_dict())
    f.gnn.load_state_dict(m.gnn.state_dict())
    f.head.load_state_dict(m.head.state_dict())
    return f
"""
old="            convhead: ConvHead, convhead_fused: ConvHeadFused, hdm: HDM}[kind](**cfg)"
new="            convhead: ConvHead, convhead_fused: ConvHeadFused, hdm: HDM,\n            resbn_gnn: ResBNGNN, resbn_gnn_fused: ResFusedGNN}[kind](**cfg)"
assert old in s, "build dict line not found"
s=s.replace(old,new)+BLOCK
open(P,"w").write(s)
print("patched OK")
