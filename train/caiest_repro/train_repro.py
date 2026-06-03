"""
train_repro.py — reproduce the caiest CNN from cooked_single.npz.
Same CNNModel; loads the single compressed npz, 95/5 split, checkpoints every epoch
(so an early checkpoint can be benchmarked before full convergence), prints val acc.

  python3 train_repro.py --epochs 20 --batch 1024 --lr 5e-4
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import CNNModel

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'cooked_single.npz'))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--ckptdir', default=os.path.join(HERE, 'log', 'checkpoint_without0'))
    a = ap.parse_args()
    os.makedirs(a.ckptdir, exist_ok=True)

    d = np.load(a.data)
    obs = torch.from_numpy(d['obs'])           # int8 (N,38,4,9)
    mask = torch.from_numpy(d['mask'])         # bool (N,235)
    act = torch.from_numpy(d['act'].astype(np.int64))
    N = obs.shape[0]
    print(f"samples={N:,}  obs{tuple(obs.shape)}  act range {int(act.min())}-{int(act.max())}", flush=True)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    nval = N // 20
    vidx, tidx = perm[:nval], perm[nval:]
    tl = DataLoader(TensorDataset(obs[tidx], mask[tidx], act[tidx]), batch_size=a.batch, shuffle=True)
    vl = DataLoader(TensorDataset(obs[vidx], mask[vidx], act[vidx]), batch_size=a.batch, shuffle=False)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNModel().to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()):,}  device={dev}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = 0.0
    for e in range(a.epochs):
        model.train()
        for i, (o, m, y) in enumerate(tl):
            inp = {'is_training': True, 'obs': {'observation': o.to(dev), 'action_mask': m.float().to(dev)}}
            loss = F.cross_entropy(model(inp), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 512 == 0:
                print(f"  ep{e} it{i}/{len(tl)} loss {loss.item():.4f}", flush=True)
        model.eval(); correct = 0
        with torch.no_grad():
            for o, m, y in vl:
                inp = {'is_training': False, 'obs': {'observation': o.to(dev), 'action_mask': m.float().to(dev)}}
                correct += (model(inp).argmax(1) == y.to(dev)).sum().item()
        acc = correct / len(vidx)
        torch.save(model.state_dict(), f"{a.ckptdir}/{e}.pkl")
        mark = ''
        if acc > best:
            best = acc; torch.save(model.state_dict(), f"{a.ckptdir}/best.pkl"); mark = '*'
        print(f"EPOCH {e+1}/{a.epochs}  val_acc={acc:.4f}  best={best:.4f} {mark}", flush=True)
    print(f"DONE best_val_acc={best:.4f} -> {a.ckptdir}/best.pkl", flush=True)
