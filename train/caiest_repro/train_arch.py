"""
train_arch.py — generalized BC trainer for the exploration architectures (38-plane data).
Copy of bc_any.py with --kind {resbn,convhead,hdm}. Trains from scratch with suit-aug + cosine,
then saves a torch-1.4-safe FUSED pkl:
  - resbn / convhead -> their fused form (resbn_fused / convhead_fused)
  - hdm -> ResFused (main head only; aux dropped) so it deploys/loads identically to resbn_fused.
For hdm the training loss = CE(main,235) + 0.3*CE(aux,type), type = ACT_TYPE_LUT[label].

  python3 train_arch.py --kind convhead --data data/cooked_single.npz \
      --channels 256 --blocks 40 --epochs 16 --out ckpt/convhead_s1.pkl --seed 1
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import build, fuse_resbn, fuse_convhead, fuse_hdm, ACT_TYPE_LUT
from suit_aug import PERMS, action_perm, fwd_action_perm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='resbn', choices=['resbn', 'convhead', 'hdm'])
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--blocks', type=int, default=24); ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--aug', type=float, default=0.8)
    ap.add_argument('--aux-w', type=float, default=0.3); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(a.data); obs, mask, act = d['obs'], d['mask'], d['act'].astype(np.int64)
    n = len(act); rng = np.random.RandomState(0); perm = rng.permutation(n)   # fixed val split (seed 0) for comparability
    nval = min(20000, n // 10); vi, ti = perm[:nval], perm[nval:]
    print(f"[{a.kind} s{a.seed}] {a.data}: {n} decisions | train {len(ti)} val {len(vi)} | ch={a.channels} blk={a.blocks}", flush=True)
    Ot = torch.from_numpy(obs); Mt = torch.from_numpy(mask); At = torch.from_numpy(act)
    rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in PERMS]
    Amaps = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    Fmaps = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    type_lut = ACT_TYPE_LUT.to(dev)
    is_hdm = (a.kind == 'hdm')
    m = build(a.kind, channels=a.channels, blocks=a.blocks).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = (len(ti) // a.bs) * a.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))

    @torch.no_grad()
    def val_acc():
        m.eval(); c = 0
        for i in range(0, len(vi), 8192):
            b = vi[i:i+8192]; o = Ot[b].to(dev); mk = Mt[b].float().to(dev)
            out = m({'is_training': False, 'obs': {'observation': o, 'action_mask': mk}})
            pr = out.argmax(1)
            c += (pr.cpu() == At[b]).sum().item()
        m.train(); return c / len(vi)

    rng2 = np.random.RandomState(a.seed + 1); best = 0.0
    for e in range(a.epochs):
        t0 = time.time(); order = rng2.permutation(len(ti)); m.train()
        for i in range(0, len(ti) - a.bs, a.bs):
            b = ti[order[i:i+a.bs]]
            o = Ot[b].to(dev); mk = Mt[b].float().to(dev); y = At[b].to(dev)
            if a.aug > 0 and rng2.random() < a.aug:
                pi = rng2.randint(1, 6); o = o[:, :, rows[pi], :]; mk = mk[:, Amaps[pi]]; y = Fmaps[pi][y]
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                if is_hdm:
                    main, aux = m({'is_training': True, 'want_aux': True,
                                   'obs': {'observation': o, 'action_mask': mk}})
                    loss = F.cross_entropy(main, y) + a.aux_w * F.cross_entropy(aux, type_lut[y])
                else:
                    out = m({'is_training': True, 'obs': {'observation': o, 'action_mask': mk}})
                    loss = F.cross_entropy(out, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        acc = val_acc(); best = max(best, acc)
        print(f"ep{e+1}/{a.epochs} val_acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    m = m.cpu().eval()
    if a.kind == 'convhead':
        fused = fuse_convhead(m)
    elif a.kind == 'hdm':
        fused = fuse_hdm(m)
    else:
        fused = fuse_resbn(m)
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE kind={a.kind} best_val_acc={best:.4f} last_val_acc={acc:.4f} -> {a.out} "
          f"(fused, ch={a.channels} blk={a.blocks})", flush=True)

if __name__ == '__main__':
    main()
