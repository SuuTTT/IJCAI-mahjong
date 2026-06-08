"""
bc_any.py — train a from-scratch behavior-cloning net on ANY (obs,mask,act) npz, then fuse to a
torch-1.4-safe ResFused pkl. Used to build diverse imitation opponents (one per 2025 finalist) for
the gauntlet. Small/fast: defaults to a 24-block resbn, suit-aug, cosine.

  python3 bc_any.py --data data/agents2025/<hash>.npz --blocks 24 --epochs 10 --out ckpt/imit_X.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn
from suit_aug import PERMS, action_perm, fwd_action_perm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--blocks', type=int, default=24); ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--aug', type=float, default=0.8)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(a.data); obs, mask, act = d['obs'], d['mask'], d['act'].astype(np.int64)
    n = len(act); rng = np.random.RandomState(0); perm = rng.permutation(n)
    nval = min(20000, n // 10); vi, ti = perm[:nval], perm[nval:]
    print(f"{a.data}: {n} decisions | train {len(ti)} val {len(vi)}", flush=True)
    Ot = torch.from_numpy(obs); Mt = torch.from_numpy(mask); At = torch.from_numpy(act)
    rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in PERMS]
    Amaps = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    Fmaps = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    m = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = (len(ti) // a.bs) * a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))

    @torch.no_grad()
    def val_acc():
        m.eval(); c = 0
        for i in range(0, len(vi), 8192):
            b = vi[i:i+8192]; o = Ot[b].to(dev); mk = Mt[b].float().to(dev)
            pr = m({'is_training': False, 'obs': {'observation': o, 'action_mask': mk}}).argmax(1)
            c += (pr.cpu() == At[b]).sum().item()
        m.train(); return c / len(vi)

    rng2 = np.random.RandomState(1); best = 0.0
    for e in range(a.epochs):
        t0 = time.time(); order = rng2.permutation(len(ti)); m.train()
        for i in range(0, len(ti) - a.bs, a.bs):
            b = ti[order[i:i+a.bs]]
            o = Ot[b].to(dev); mk = Mt[b].float().to(dev); y = At[b].to(dev)
            if a.aug > 0 and rng2.random() < a.aug:
                pi = rng2.randint(1, 6); o = o[:, :, rows[pi], :]; mk = mk[:, Amaps[pi]]; y = Fmaps[pi][y]
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                loss = F.cross_entropy(m({'is_training': True, 'obs': {'observation': o, 'action_mask': mk}}), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        acc = val_acc(); best = max(best, acc)
        print(f"ep{e+1}/{a.epochs} val_acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    fused = fuse_resbn(m.cpu().eval())
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE best_val_acc={best:.4f} -> {a.out} (fused, blocks={a.blocks})", flush=True)

if __name__ == '__main__':
    main()
