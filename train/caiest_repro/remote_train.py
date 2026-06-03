"""
remote_train.py — self-contained single-arch trainer for the fleet SL exploration.
Runs on a free GPU box (needs only torch + numpy + models_explore.py + cooked_single.npz).
Trains one architecture, prints val acc/epoch, saves the best state_dict to --out.

  python3 remote_train.py --kind attn --cfg '{"d_model":192,"layers":6,"heads":8}' \
      --epochs 14 --batch 1024 --out attn.pkl --data cooked_single.npz
The 'base' kind trains the caiest CNNModel (reference) if model.py is present.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

def build_model(kind, cfg):
    if kind == 'base':
        from model import CNNModel; return CNNModel()
    from models_explore import build; return build(kind, **cfg)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True)
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--data', default='cooked_single.npz')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(a.data)
    obs = torch.from_numpy(d['obs']); mask = torch.from_numpy(d['mask']); act = torch.from_numpy(d['act'].astype(np.int64))
    N = obs.shape[0]; ds = TensorDataset(obs, mask, act)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    nval = N // 20
    tl = DataLoader(Subset(ds, perm[nval:].tolist()), batch_size=a.batch, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(Subset(ds, perm[:nval].tolist()), batch_size=a.batch, shuffle=False, num_workers=2)
    model = build_model(a.kind, cfg).to(dev)
    np_ = sum(p.numel() for p in model.parameters())
    print(f"kind={a.kind} cfg={cfg} params={np_/1e6:.1f}M N={N} dev={dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))
    best = -1.0
    for e in range(a.epochs):
        model.train(); t0 = time.time()
        for o, m, y in tl:
            o = o.to(dev, non_blocking=True); m = m.float().to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                loss = F.cross_entropy(model({'is_training': True, 'obs': {'observation': o, 'action_mask': m}}), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); correct = 0
        with torch.no_grad():
            for o, m, y in vl:
                o = o.to(dev); m = m.float().to(dev); y = y.to(dev)
                correct += (model({'is_training': False, 'obs': {'observation': o, 'action_mask': m}}).argmax(1) == y).sum().item()
        acc = correct / nval
        if acc > best:
            best = acc
            torch.save(model.state_dict(), a.out, _use_new_zipfile_serialization=False)
        print(f"EPOCH {e+1}/{a.epochs} val_acc={acc:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE kind={a.kind} best_val_acc={best:.4f} -> {a.out}", flush=True)
