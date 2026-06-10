"""
value_head.py — train V(obs) ~ the acting seat's FINAL duplicate score, on the FROZEN distill100b
trunk (stem+body of ResFused) + a small head (GAP -> 128 -> 64 -> 1). Foundation for test-time
search (policy-top-k + value rerank) AND a standalone deal-in-awareness signal.

Time-box gate: if val Pearson r < ~0.25 the position carries too little outcome signal at this data
scale and the search bet should be dropped. Target scale: score/48 (typical win ~ +0.5..+2).

  python3 value_head.py --base ckpt/distill100b_fused.pkl --data data/value_recent.npz \
      --out ckpt/value_head.pkl [--unfreeze 4] [--epochs 8]
Saves {'head': state_dict, 'unfreeze': k, 'blocks': n} (head + optionally last-k fine-tuned blocks).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from models_explore import ResFused


class ValueNet(nn.Module):
    def __init__(self, fused, unfreeze=0):
        super().__init__()
        self.stem, self.body = fused.stem, fused.body
        for p in self.parameters():
            p.requires_grad_(False)
        if unfreeze > 0:                       # optionally fine-tune the last k blocks
            for blk in list(self.body)[-unfreeze:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        ch = self.stem.out_channels
        self.head = nn.Sequential(nn.Linear(ch, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs):                    # obs (B,38,4,9) float
        x = torch.relu(self.stem(obs))
        x = self.body(x)
        x = x.mean(dim=(2, 3))                 # GAP -> (B,ch)
        return self.head(x).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--blocks', type=int, default=40); ap.add_argument('--unfreeze', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(a.data)
    obs = z['obs'].astype(np.float32); y = (z['score'].astype(np.float32) / 48.0)
    rng = np.random.RandomState(0); idx = rng.permutation(len(y))
    nval = max(500, len(idx) // 10); vi, ti = idx[:nval], idx[nval:]
    print(f"value data: {len(y)} decisions, train {len(ti)} val {nval}, y mean {y.mean():+.3f} std {y.std():.3f}", flush=True)
    fused = ResFused(channels=128, blocks=a.blocks)
    fused.load_state_dict(torch.load(a.base, map_location='cpu'))
    net = ValueNet(fused, unfreeze=a.unfreeze).to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr)
    vO = torch.from_numpy(obs[vi]); vY = y[vi]
    def val_stats():
        net.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(vi), 1024):
                preds.append(net(vO[i:i+1024].to(dev)).cpu().numpy())
        p = np.concatenate(preds)
        r = float(np.corrcoef(p, vY)[0, 1]); mae = float(np.abs(p - vY).mean())
        return r, mae
    n = len(ti); r0, m0 = val_stats()
    print(f"init: val r={r0:+.3f} mae={m0:.3f}", flush=True)
    for ep in range(1, a.epochs + 1):
        net.train(); perm = np.random.RandomState(ep).permutation(n)
        for i in range(0, n, a.bs):
            b = ti[perm[i:i+a.bs]]
            ob = torch.from_numpy(obs[b]).to(dev); yy = torch.from_numpy(y[b]).to(dev)
            loss = nn.functional.smooth_l1_loss(net(ob), yy)
            opt.zero_grad(); loss.backward(); opt.step()
        r, mae = val_stats()
        print(f"ep {ep}/{a.epochs}: val r={r:+.3f} mae={mae:.3f}", flush=True)
    # save the FULL self-contained value net (stem+body+head) for deploy-side search
    torch.save({k: v.cpu() for k, v in net.state_dict().items()},
               a.out, _use_new_zipfile_serialization=False)
    print(f"DONE val_r={r:+.3f} -> {a.out} (full net; gate: search only if r >= ~0.25)", flush=True)


if __name__ == '__main__':
    main()
