"""
supervised_v2.py — stronger SL training on the 5.87M official data (the WINNING axis).
The original resbn40 used a basic recipe (plain Adam, few epochs, no augmentation). This adds:
AdamW + weight decay, cosine LR schedule, ON-THE-FLY suit-permutation augmentation (the 3 number
suits are symmetric in MCR -> free 6x regularization), AMP, more epochs. Measured by held-out
val_acc (clean, non-noisy) — and we gauntlet the result vs resbn40 afterward.

  python3 supervised_v2.py --warm arch_ck/explore/resbn40.pkl --epochs 6 --out arch_ck/explore/resbn40_sl2.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from models_explore import build
from suit_aug import PERMS, action_perm, fwd_action_perm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warm', default='')                 # warm-start ckpt ('' = from scratch)
    ap.add_argument('--kind', default='resbn')            # any models_explore build() kind (resbn/cnn/attn/...)
    ap.add_argument('--blocks', type=int, default=40); ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--aug', type=float, default=0.8)      # prob of applying a random suit perm per batch
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ddir = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(os.path.join(ddir, 'cooked_obs.npy')):   # memmap triplet (small-RAM boxes)
        obs = np.load(os.path.join(ddir, 'cooked_obs.npy'), mmap_mode='r')
        mask = np.load(os.path.join(ddir, 'cooked_mask.npy'), mmap_mode='r')
        act = np.load(os.path.join(ddir, 'cooked_act.npy'), mmap_mode='r')
    else:
        d = np.load(os.path.join(ddir, 'cooked_single.npz'))
        obs, mask, act = d['obs'], d['mask'], d['act'].astype(np.int64)
    n = len(act); rng = np.random.RandomState(0); perm = rng.permutation(n)
    nval = 200000; vi, ti = perm[:nval], perm[nval:]
    print(f"data {n:,} | train {len(ti):,} val {len(vi):,}", flush=True)
    Ot = lambda b: torch.from_numpy(np.ascontiguousarray(obs[b]))
    Mt = lambda b: torch.from_numpy(np.ascontiguousarray(mask[b]))
    At = lambda b: torch.from_numpy(np.ascontiguousarray(act[b]).astype(np.int64))
    # precompute suit-perm index maps on device
    rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in PERMS]
    Amaps = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]   # new_mask = old[A]
    Fmaps = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]

    m = build(a.kind, channels=a.channels, blocks=a.blocks).to(dev)
    if a.warm:
        m.load_state_dict(torch.load(a.warm, map_location='cpu')); print(f"warm-start {a.warm}", flush=True)

    @torch.no_grad()
    def val_acc():
        m.eval(); correct = 0
        for i in range(0, len(vi), 8192):
            b = np.sort(vi[i:i + 8192])                     # sorted = sequential-ish memmap reads
            o = Ot(b).to(dev); mk = Mt(b).float().to(dev)
            pr = m({'is_training': False, 'obs': {'observation': o, 'action_mask': mk}}).argmax(1)
            correct += (pr.cpu() == At(b)).sum().item()
        m.train(); return correct / len(vi)

    base_acc = val_acc(); print(f"[val] warm/baseline acc = {base_acc:.4f}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = (len(ti) // a.bs) * a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))
    best = base_acc; rng2 = np.random.RandomState(1)
    for e in range(a.epochs):
        t0 = time.time(); order = rng2.permutation(len(ti)); m.train()
        for i in range(0, len(ti) - a.bs, a.bs):
            b = np.sort(ti[order[i:i + a.bs]])              # sorted within batch (order is still random)
            o = Ot(b).to(dev); mk = Mt(b).float().to(dev); y = At(b).to(dev)
            if a.aug > 0 and rng2.random() < a.aug:
                pi = rng2.randint(1, 6)                          # a non-identity suit permutation
                o = o[:, :, rows[pi], :]; mk = mk[:, Amaps[pi]]; y = Fmaps[pi][y]
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                loss = F.cross_entropy(m({'is_training': True, 'obs': {'observation': o, 'action_mask': mk}}), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        acc = val_acc()
        if acc > best:
            best = acc; torch.save(m.state_dict(), a.out, _use_new_zipfile_serialization=False)
            tag = 'NEW BEST -> saved'
        else:
            tag = ''
        print(f"ep{e+1}/{a.epochs} val_acc={acc:.4f} (base {base_acc:.4f}, best {best:.4f}) {tag} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE best_val_acc={best:.4f} (baseline {base_acc:.4f}) -> {a.out}", flush=True)

if __name__ == '__main__':
    main()
