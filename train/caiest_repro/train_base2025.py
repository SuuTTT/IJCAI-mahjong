"""
train_base2025.py (P4) — stronger SL base on official + 2025-final data. Loads the official cooked
memmap (5.87M) and an EXTRA on-distribution npz (2025 finalists' decisions), over-weights the extra
to ~extra_frac of each batch (the 2025 data is the actual final distribution). Modern recipe
(AdamW + cosine + suit-aug + AMP). Saves best-val ResBNCNN, then fuses to a torch-1.4-safe pkl.

  python3 train_base2025.py --extra data/final2025_all.npz --extra-frac 0.3 --blocks 40 \
      --epochs 6 --out /root/mahjong/ckpt/base2025.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn
from suit_aug import PERMS, action_perm, fwd_action_perm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extra', required=True); ap.add_argument('--extra-frac', dest='ef', type=float, default=0.3)
    ap.add_argument('--warm', default=''); ap.add_argument('--blocks', type=int, default=40); ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--wd', type=float, default=1e-4); ap.add_argument('--aug', type=float, default=0.8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ddir = os.path.join(os.path.dirname(__file__), 'data')
    o_off = np.load(os.path.join(ddir, 'cooked_obs.npy'), mmap_mode='r')
    m_off = np.load(os.path.join(ddir, 'cooked_mask.npy'), mmap_mode='r')
    a_off = np.load(os.path.join(ddir, 'cooked_act.npy'), mmap_mode='r')
    ex = np.load(a.extra); xo, xm, xa = ex['obs'], ex['mask'], ex['act'].astype(np.int64)
    no, nx = len(a_off), len(xa)
    rng = np.random.RandomState(0)
    voff = rng.permutation(no)[:100000]; vx = rng.permutation(nx)[:30000]
    print(f"official {no:,} + extra {nx:,} (extra-frac {a.ef}) | val off {len(voff)} extra {len(vx)}", flush=True)
    Xo = torch.from_numpy(xo); Xm = torch.from_numpy(xm); Xa = torch.from_numpy(xa)
    rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in PERMS]
    Am = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    Fm = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    m = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    if a.warm: m.load_state_dict(torch.load(a.warm, map_location='cpu'))

    @torch.no_grad()
    def acc(idx, src):
        m.eval(); c = 0
        for i in range(0, len(idx), 8192):
            b = np.sort(idx[i:i + 8192])
            if src == 'off':
                o = torch.from_numpy(np.ascontiguousarray(o_off[b])); mk = torch.from_numpy(np.ascontiguousarray(m_off[b])).float(); y = torch.from_numpy(np.ascontiguousarray(a_off[b]).astype(np.int64))
            else:
                o = Xo[b]; mk = Xm[b].float(); y = Xa[b]
            pr = m({'is_training': False, 'obs': {'observation': o.to(dev), 'action_mask': mk.to(dev)}}).argmax(1)
            c += (pr.cpu() == y).sum().item()
        m.train(); return c / len(idx)

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    spe = (no // a.bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spe * a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))
    r2 = np.random.RandomState(1); best = 0.0
    for e in range(a.epochs):
        t0 = time.time(); order = r2.permutation(no); m.train()
        for i in range(0, no - a.bs, a.bs):
            if r2.random() < a.ef:                                   # draw an EXTRA (2025) batch
                bi = r2.randint(0, nx, size=a.bs)
                o = Xo[bi].to(dev); mk = Xm[bi].float().to(dev); y = Xa[bi].to(dev)
            else:                                                    # official batch (sorted for memmap)
                b = np.sort(order[i:i + a.bs])
                o = torch.from_numpy(np.ascontiguousarray(o_off[b])).to(dev)
                mk = torch.from_numpy(np.ascontiguousarray(m_off[b])).float().to(dev)
                y = torch.from_numpy(np.ascontiguousarray(a_off[b]).astype(np.int64)).to(dev)
            if a.aug > 0 and r2.random() < a.aug:
                pi = r2.randint(1, 6); o = o[:, :, rows[pi], :]; mk = mk[:, Am[pi]]; y = Fm[pi][y]
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                loss = F.cross_entropy(m({'is_training': True, 'obs': {'observation': o, 'action_mask': mk}}), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ao = acc(voff, 'off'); ax = acc(vx, 'extra'); comb = 0.5 * ao + 0.5 * ax
        tag = ''
        if comb > best:
            best = comb; torch.save(m.state_dict(), a.out, _use_new_zipfile_serialization=False); tag = 'NEW BEST'
        print(f"ep{e+1}/{a.epochs} val_off={ao:.4f} val_2025={ax:.4f} comb={comb:.4f} {tag} ({time.time()-t0:.0f}s)", flush=True)
    # fuse best -> deployable
    m.load_state_dict(torch.load(a.out, map_location='cpu'))
    fused = fuse_resbn(m.cpu().eval())
    fout = a.out.replace('.pkl', '_fused.pkl')
    torch.save(fused.state_dict(), fout, _use_new_zipfile_serialization=False)
    print(f"DONE best_comb={best:.4f} -> {a.out} | fused -> {fout}", flush=True)

if __name__ == '__main__':
    main()
