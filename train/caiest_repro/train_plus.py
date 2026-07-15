"""
train_plus.py — train ResBNCNNP(in_planes=38+P) on cooked data with AUGMENTED feature planes.

Base obs come from cooked_obs.npy (N,38,4,9). Extra planes per --sets:
  A,C  : derived on-the-fly from the base obs batch (fast numpy, vectorised).
  B    : read from a precomputed memmap (data/planeB.npy, (N,4,4,9) float32); built by cook_planeB.py.
Suit-aug: rows[pi] permutes dim-2 (suit rows 0,1,2) of the FULL (38+P) tensor — all featplus planes
are per-suit-row so this stays label-preserving (honors row fixed). AdamW+cosine+AMP, 95/5 split.

  CUDA_VISIBLE_DEVICES=0 python3 train_plus.py --sets A --epochs 16 --out ckpt/featA.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_plus import ResBNCNNP
from suit_aug import PERMS, action_perm, fwd_action_perm
import featplus

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, 'data')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', required=True)
    ap.add_argument('--blocks', type=int, default=40); ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=16); ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--aug', type=float, default=0.8); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--planeB', default=os.path.join(DDIR, 'planeB.npy'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda'
    sets = a.sets
    P = featplus.total_extra(sets)
    o = np.load(os.path.join(DDIR, 'cooked_obs.npy'), mmap_mode='r')
    m = np.load(os.path.join(DDIR, 'cooked_mask.npy'), mmap_mode='r')
    ac = np.load(os.path.join(DDIR, 'cooked_act.npy'), mmap_mode='r')
    N = len(ac)
    planeB = None
    if 'B' in sets:
        planeB = np.load(a.planeB, mmap_mode='r')
        assert planeB.shape[0] == N, f"planeB N {planeB.shape[0]} != {N}"
    rng = np.random.RandomState(a.seed); perm = rng.permutation(N)
    nval = N // 20; vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    print(f"sets={sets} P={P} in_planes={38+P}  samples {N:,} train {len(tidx):,} val {len(vidx):,}", flush=True)
    rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in PERMS]
    Am = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    Fm = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    torch.manual_seed(a.seed); net = ResBNCNNP(in_planes=38 + P, channels=a.channels, blocks=a.blocks).to(dev)
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    spe = len(tidx) // a.bs; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spe * a.epochs)
    scaler = torch.cuda.amp.GradScaler()

    def build_obs(idx):
        base = np.ascontiguousarray(o[idx]).astype(np.float32)  # (B,38,4,9)
        # canonical plane order: base, A, B, C
        out = [base]
        if 'A' in sets: out.append(featplus.planes_A(base))
        if 'B' in sets: out.append(np.ascontiguousarray(planeB[idx]).astype(np.float32))
        if 'C' in sets: out.append(featplus.planes_C(base))
        return np.concatenate(out, axis=1)

    @torch.no_grad()
    def val():
        net.eval(); c = 0
        for i in range(0, len(vidx), 8192):
            b = vidx[i:i + 8192]
            ob = torch.from_numpy(build_obs(b)).to(dev)
            mk = torch.from_numpy(np.ascontiguousarray(m[b])).float().to(dev)
            y = torch.from_numpy(np.ascontiguousarray(ac[b]).astype(np.int64))
            pr = net({'is_training': False, 'obs': {'observation': ob, 'action_mask': mk}}).argmax(1).cpu()
            c += (pr == y).sum().item()
        net.train(); return c / len(vidx)

    r2 = np.random.RandomState(a.seed + 1); best = 0.0; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for e in range(a.epochs):
        t0 = time.time(); order = r2.permutation(len(tidx)); net.train()
        for i in range(0, len(tidx) - a.bs, a.bs):
            b = np.sort(tidx[order[i:i + a.bs]])
            ob = torch.from_numpy(build_obs(b)).to(dev)
            mk = torch.from_numpy(np.ascontiguousarray(m[b])).float().to(dev)
            y = torch.from_numpy(np.ascontiguousarray(ac[b]).astype(np.int64)).to(dev)
            if a.aug > 0 and r2.random() < a.aug:
                pi = r2.randint(1, 6); ob = ob[:, :, rows[pi], :]; mk = mk[:, Am[pi]]; y = Fm[pi][y]
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(net({'is_training': True, 'obs': {'observation': ob, 'action_mask': mk}}), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if i // a.bs % 500 == 0: print(f"  ep{e} it{i//a.bs}/{spe} loss {loss.item():.4f}", flush=True)
        v = val(); tag = ''
        if v > best: best = v; torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False); tag = 'NEW BEST'
        print(f"ep{e+1}/{a.epochs} val={v:.4f} {tag} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE sets={sets} best_val={best:.4f} -> {a.out}", flush=True)

if __name__ == '__main__':
    main()
