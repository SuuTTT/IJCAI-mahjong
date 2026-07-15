"""
finetune_bc.py — finetune an existing ResBNCNN checkpoint (BN form) with BC on new data,
optionally MIXING a teacher npz with the base (full-action) data so claims/hu aren't forgotten.
Saves a fused (torch-1.4-safe) pkl. Reusable for the strong-teacher (mythos/finalist) collection.

  python3 finetune_bc.py --init ckpt/big256x40_s0.pkl --data data/cooked_single.npz \
     --data2 /root/assets/union_chun_top30.npz --mix 0.3 --channels 256 --blocks 40 \
     --steps 8000 --lr 1e-4 --out ckpt/ft_mix256.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn
from suit_aug import PERMS, action_perm, fwd_action_perm

def load_npz(p):
    d = np.load(p); return d['obs'], d['mask'], d['act'].astype(np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--data', required=True)          # primary full-action data (e.g. cooked_single)
    ap.add_argument('--data2', default='')            # teacher data (e.g. union_chun_top30), optional
    ap.add_argument('--mix', type=float, default=0.0) # fraction of batches drawn from data2
    ap.add_argument('--channels', type=int, default=256); ap.add_argument('--blocks', type=int, default=40)
    ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--aug', type=float, default=0.8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    O1, M1, A1 = load_npz(a.data)
    Ot1, Mt1, At1 = torch.from_numpy(O1), torch.from_numpy(M1), torch.from_numpy(A1)
    use2 = bool(a.data2) and a.mix > 0
    if use2:
        O2, M2, A2 = load_npz(a.data2)
        Ot2, Mt2, At2 = torch.from_numpy(O2), torch.from_numpy(M2), torch.from_numpy(A2)
        print(f"mix: {a.mix:.2f} from {a.data2} ({len(A2)}) / rest from {a.data} ({len(A1)})", flush=True)
    else:
        print(f"finetune on {a.data} ({len(A1)}) only", flush=True)
    m = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    m.load_state_dict(torch.load(a.init, map_location='cpu'))
    print(f"loaded init {a.init}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))
    rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in PERMS]
    Amaps = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    Fmaps = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    rng = np.random.RandomState(1); n1 = len(At1); n2 = len(At2) if use2 else 0
    m.train(); t0 = time.time()
    for s in range(a.steps):
        if use2 and rng.random() < a.mix:
            b = rng.randint(0, n2, a.bs); o = Ot2[b].to(dev); mk = Mt2[b].float().to(dev); y = At2[b].to(dev)
        else:
            b = rng.randint(0, n1, a.bs); o = Ot1[b].to(dev); mk = Mt1[b].float().to(dev); y = At1[b].to(dev)
        if a.aug > 0 and rng.random() < a.aug:
            pi = rng.randint(1, 6); o = o[:, :, rows[pi], :]; mk = mk[:, Amaps[pi]]; y = Fmaps[pi][y]
        with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
            loss = F.cross_entropy(m({'is_training': True, 'obs': {'observation': o, 'action_mask': mk}}), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if (s + 1) % 1000 == 0:
            print(f"step{s+1}/{a.steps} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    fused = fuse_resbn(m.cpu().eval())
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE -> {a.out} (fused)", flush=True)

if __name__ == '__main__':
    main()
