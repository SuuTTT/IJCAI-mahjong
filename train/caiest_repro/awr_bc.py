"""
awr_bc.py — Advantage-Weighted BC. Finetune a ResBNCNN (BN form) toward the leaders' decisions,
weighting each leader sample by a function of the DEAL OUTCOME (reward = seat's MCR score that deal).

  weight modes (on leader samples only):
    exp  : w = clip(exp(beta * (reward - base)/scale), wlo, whi)        [AWR-style]
    lin  : w = clip(1 + reward/K, wlo, whi)                              [simple linear]
    pos  : w = 1 if reward>0 else gamma   (keep only winning decisions strongly)
  base/scale default to data mean/std. Base full-action mix samples always weight 1.

Saves a fused (torch-1.4-safe) pkl, same format finetune_bc uses (loadable by frontier_gate
with --cand-kind resbn_fused --cand-cfg channels=128,blocks=40).

  python3 awr_bc.py --init /root/assets/moyu_bn_128x40.pkl \
     --leader data/teachers/leaders_outcome.npz \
     --base /root/realfield_build/base_256x40.npz --mix 0.25 \
     --mode exp --beta 1.0 --steps 6000 --out ckpt/awr/moyu_exp_b1.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, "/root/IJCAI-mahjong/train/caiest_repro")
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn
from suit_aug import PERMS, action_perm, fwd_action_perm

def compute_weights(reward, mode, beta, K, scale, base, wlo, whi, gamma):
    r = reward.astype(np.float64)
    if base is None: base = r.mean()
    if scale is None or scale <= 0: scale = r.std() + 1e-6
    if mode == "exp":
        w = np.exp(beta * (r - base) / scale)
    elif mode == "lin":
        w = 1.0 + r / K
    elif mode == "pos":
        w = np.where(r > 0, 1.0, gamma)
    else:
        raise ValueError(mode)
    w = np.clip(w, wlo, whi)
    return w.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--leader", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--mix", type=float, default=0.0)   # frac of batches from base full-action data
    ap.add_argument("--channels", type=int, default=128); ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--aug", type=float, default=0.8)
    ap.add_argument("--mode", default="exp", choices=["exp", "lin", "pos"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=32.0)
    ap.add_argument("--scale", type=float, default=-1.0)
    ap.add_argument("--base_r", type=float, default=1e9)  # sentinel -> use data mean
    ap.add_argument("--wlo", type=float, default=0.25); ap.add_argument("--whi", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    d = np.load(a.leader)
    OL, ML, AL = d["obs"], d["mask"], d["act"].astype(np.int64)
    RL = d["reward"].astype(np.float32)
    scale = None if a.scale <= 0 else a.scale
    base = None if a.base_r >= 1e8 else a.base_r
    W = compute_weights(RL, a.mode, a.beta, a.K, scale, base, a.wlo, a.whi, a.gamma)
    print(f"leader N={len(AL)} mode={a.mode} beta={a.beta} K={a.K} "
          f"weight: min={W.min():.3f} max={W.max():.3f} mean={W.mean():.3f} "
          f"(corr w,reward={np.corrcoef(W, RL)[0,1]:.3f})", flush=True)
    OLt, MLt, ALt, Wt = (torch.from_numpy(OL), torch.from_numpy(ML),
                         torch.from_numpy(AL), torch.from_numpy(W))

    use2 = bool(a.base) and a.mix > 0
    if use2:
        db = np.load(a.base)
        OB, MB, AB = db["obs"], db["mask"], db["act"].astype(np.int64)
        OBt, MBt, ABt = torch.from_numpy(OB), torch.from_numpy(MB), torch.from_numpy(AB)
        print(f"mix {a.mix:.2f} from base {a.base} (N={len(AB)})", flush=True)

    m = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    sd = torch.load(a.init, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    m.load_state_dict(sd)
    print(f"loaded init {a.init}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in PERMS]
    Amaps = [torch.tensor(action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    Fmaps = [torch.tensor(fwd_action_perm(p), device=dev, dtype=torch.long) for p in PERMS]
    rng = np.random.RandomState(1); nL = len(ALt); nB = len(ABt) if use2 else 0
    m.train(); t0 = time.time()
    for s in range(a.steps):
        from_base = use2 and rng.random() < a.mix
        if from_base:
            b = rng.randint(0, nB, a.bs)
            o = OBt[b].to(dev); mk = MBt[b].float().to(dev); y = ABt[b].to(dev)
            w = torch.ones(a.bs, device=dev)
        else:
            b = rng.randint(0, nL, a.bs)
            o = OLt[b].to(dev); mk = MLt[b].float().to(dev); y = ALt[b].to(dev)
            w = Wt[b].to(dev)
        if a.aug > 0 and rng.random() < a.aug:
            pi = rng.randint(1, 6); o = o[:, :, rows[pi], :]; mk = mk[:, Amaps[pi]]; y = Fmaps[pi][y]
        with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            logits = m({"is_training": True, "obs": {"observation": o, "action_mask": mk}})
            ce = F.cross_entropy(logits, y, reduction="none")
            loss = (w * ce).sum() / (w.sum() + 1e-6)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if (s + 1) % 1000 == 0:
            print(f"step{s+1}/{a.steps} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    fused = fuse_resbn(m.cpu().eval())
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE -> {a.out} (fused)", flush=True)

if __name__ == "__main__":
    main()
