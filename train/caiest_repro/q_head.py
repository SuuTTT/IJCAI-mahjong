"""
q_head.py — action-conditioned outcome model Q(s,a) ~ final duplicate score, for test-time
DISCARD RERANKING (the minimal exceed-the-teacher search). Trunk = distill100b ResFused with the
last-k blocks unfrozen (value_head.py showed frozen r=0.27 vs unfreeze-4 r=0.72), head =
GAP(128) ++ act-embedding(64) -> MLP -> scalar. Trained on (obs, TAKEN action, score/48) from
top-30 seats — on-policy Q, so deploy-side reranking must stay among policy top-k (near-policy)
actions to avoid OOD extrapolation.

12x suit/reflection augmentation (actions remap correctly via suit_aug.augment).

  python3 q_head.py --base ckpt/distill100b_fused.pkl --data data/value_recent.npz \
      --out ckpt/qnet.pkl [--unfreeze 4] [--epochs 4]
Saves the FULL state_dict (trunk+emb+head) — a self-contained Q net for the rerank prototype.
Gate printed at the end: action-discrimination AUC-like check (does Q rank the taken action of
WINNING seats above the taken action of LOSING seats at matched positions? proxy: corr + by-stage r).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from models_explore import ResFused
from suit_aug import augment


class QNet(nn.Module):
    def __init__(self, fused, unfreeze=4):
        super().__init__()
        self.stem, self.body = fused.stem, fused.body
        for p in self.parameters():
            p.requires_grad_(False)
        for blk in list(self.body)[-unfreeze:] if unfreeze else []:
            for p in blk.parameters():
                p.requires_grad_(True)
        ch = self.stem.out_channels
        self.emb = nn.Embedding(235, 64)
        self.head = nn.Sequential(nn.Linear(ch + 64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def feats(self, obs):
        x = torch.relu(self.stem(obs))
        return self.body(x).mean(dim=(2, 3))   # (B,ch)

    def forward(self, obs, act):               # obs (B,38,4,9) float, act (B,) long
        f = self.feats(obs)
        return self.head(torch.cat([f, self.emb(act)], 1)).squeeze(1)

    def q_all(self, obs, acts):                # one trunk pass, many actions: obs (1,...), acts (K,)
        f = self.feats(obs).expand(len(acts), -1)
        return self.head(torch.cat([f, self.emb(acts)], 1)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--blocks', type=int, default=40); ap.add_argument('--unfreeze', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(a.data)
    obs, mask, act = z['obs'], z['mask'], z['act'].astype(np.int64)
    y = z['score'].astype(np.float32) / 48.0
    rng = np.random.RandomState(0); idx = rng.permutation(len(y))
    nval = max(500, len(idx) // 10); vi, ti = idx[:nval], idx[nval:]
    co, cm, ca = augment(obs[ti], mask[ti], act[ti])          # 12x (actions remapped)
    cy = np.tile(y[ti], 12)
    print(f"Q data: {len(y)} -> aug {len(ca)} train, {nval} val; y std {y.std():.3f}", flush=True)
    fused = ResFused(channels=128, blocks=a.blocks)
    fused.load_state_dict(torch.load(a.base, map_location='cpu'))
    net = QNet(fused, unfreeze=a.unfreeze).to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad] + list(net.emb.parameters()) + list(net.head.parameters()), lr=a.lr)
    vO = torch.from_numpy(obs[vi].astype(np.float32)); vA = torch.from_numpy(act[vi]); vY = y[vi]
    def val_r():
        net.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(vi), 1024):
                preds.append(net(vO[i:i+1024].to(dev), vA[i:i+1024].to(dev)).cpu().numpy())
        p = np.concatenate(preds)
        return float(np.corrcoef(p, vY)[0, 1])
    n = len(ca)
    for ep in range(1, a.epochs + 1):
        net.train(); perm = np.random.RandomState(ep).permutation(n)
        for i in range(0, n, a.bs):
            b = perm[i:i+a.bs]
            ob = torch.from_numpy(co[b].astype(np.float32)).to(dev)
            ab = torch.from_numpy(ca[b]).to(dev); yy = torch.from_numpy(cy[b]).to(dev)
            loss = nn.functional.smooth_l1_loss(net(ob, ab), yy)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"ep {ep}/{a.epochs}: val r={val_r():+.3f}", flush=True)
    torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE -> {a.out}", flush=True)


if __name__ == '__main__':
    main()
