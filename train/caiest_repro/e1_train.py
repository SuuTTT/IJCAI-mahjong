"""
e1_train.py — E1 from-scratch BC trainer for the over-claiming study (IEEE ToG).

Mirrors train_base_official.py's recipe (AdamW + cosine + on-GPU suit-aug + AMP, 95/5 split),
but with a FIXED STEP BUDGET (so data-fraction / composition arms see the same #updates, fair
for capacity & composition comparison), configurable channels/blocks/seed, optional data fraction,
and an optional npz training set (for the top-only arm). Saves a FUSED (torch-1.4-safe) ResFused
state_dict for downstream gating, plus best BN state_dict alongside (.bn.pkl).

Data sources:
  --data full         -> mmap the official raw cooked_obs/mask/act .npy (the ~5.87M mixed set)
  --data <path.npz>   -> load an npz with obs,mask,act (e.g. a top-only expert subset)

  CUDA_VISIBLE_DEVICES=0 python3 e1_train.py --channels 128 --blocks 40 --steps 30000 \
     --seed 0 --data full --out ckpt/e1/full_128x40_s0.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn
from suit_aug import PERMS, action_perm, fwd_action_perm

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--aug", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="full")         # "full" -> mmap raw; else path to npz
    ap.add_argument("--frac", type=float, default=1.0) # fraction of TRAIN rows to use
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    # Load fully into RAM (cooked_single.npz == raw .npy data, verified). In-RAM random
    # indexing is far faster than mmap fancy-indexing and avoids 4-way disk thrash when
    # several GPU jobs run concurrently. obs int8 ~8GB; box has 503GB.
    src = os.path.join(DDIR, "cooked_single.npz") if a.data == "full" else a.data
    d = np.load(src)
    o, m, ac = d["obs"], d["mask"], d["act"].astype(np.int64)
    N = len(ac)
    # split is SEED-FIXED so val set is identical across the matrix (comparable val_acc);
    # train-subsampling (frac) uses the run seed so different seeds see different subsets.
    rng = np.random.RandomState(12345); perm = rng.permutation(N)
    nval = min(50000, N // 20); vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    if a.frac < 1.0:
        r = np.random.RandomState(1000 + a.seed)
        keep = int(len(tidx) * a.frac); tidx = tidx[r.permutation(len(tidx))[:keep]]
    print(f"data={a.data} N={N:,} train={len(tidx):,} (frac={a.frac}) val={len(vidx):,} "
          f"ch={a.channels} blk={a.blocks} seed={a.seed} steps={a.steps}", flush=True)

    rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in PERMS]
    Am = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    Fm = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    net = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    scaler = torch.cuda.amp.GradScaler()

    def fetch(idx):
        idx = np.sort(idx)
        ob = torch.from_numpy(np.ascontiguousarray(o[idx])).to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(m[idx])).float().to(dev)
        y = torch.from_numpy(np.ascontiguousarray(ac[idx]).astype(np.int64)).to(dev)
        return ob, mk, y

    @torch.no_grad()
    def val():
        net.eval(); c = 0
        for i in range(0, len(vidx), 8192):
            b = vidx[i:i + 8192]
            ob, mk, y = fetch(b)
            pr = net({"is_training": False, "obs": {"observation": ob, "action_mask": mk}}).argmax(1)
            c += (pr == y).sum().item()
        net.train(); return c / len(vidx)

    r2 = np.random.RandomState(1 + a.seed); best = 0.0; nt = len(tidx)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    bn_out = a.out[:-4] + ".bn.pkl"
    t0 = time.time(); net.train()
    for s in range(a.steps):
        b = tidx[r2.randint(0, nt, a.bs)]
        ob, mk, y = fetch(b)
        if a.aug > 0 and r2.random() < a.aug:
            pi = r2.randint(1, 6); ob = ob[:, :, rows[pi], :]; mk = mk[:, Am[pi]]; y = Fm[pi][y]
        with torch.cuda.amp.autocast():
            loss = F.cross_entropy(net({"is_training": True, "obs": {"observation": ob, "action_mask": mk}}), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if s % 2000 == 0:
            v = val()
            if v > best:
                best = v; torch.save(net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} val {v:.4f} best {best:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    v = val()
    if v > best:
        best = v; torch.save(net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
    # fuse the BEST BN snapshot and save the fused pkl (used for gating + claimrate)
    bestnet = ResBNCNN(channels=a.channels, blocks=a.blocks)
    bestnet.load_state_dict(torch.load(bn_out, map_location="cpu")); bestnet.eval()
    fused = fuse_resbn(bestnet)
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE best_val={best:.4f} -> {a.out} (fused) + {bn_out} (bn)", flush=True)


if __name__ == "__main__":
    main()
