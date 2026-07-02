"""
e13_kd_train.py — KD-from-ensemble ENHANCED-recipe 128x40 BC trainer (deployable size) for the aug/regularization
push. Same data + SAME seed-fixed val split as e1_train.py (so val_acc is directly comparable
to bn128s1's 0.887), but with:
  * augmentation = suit-perm (6, existing) x rank-reflection (2, verified) x dragon-perm (6, verified)
    composed on-GPU -> up to 72x label-preserving effective data.
  * label smoothing (over LEGAL actions only, so masked -inf classes never contribute).
  * EMA of weights (decay 0.999) — eval + save the EMA snapshot.
  * warmup + cosine schedule, longer budget (default 130k), early-stop on best EMA val.
  * mild weight decay bump.
Saves a FUSED (torch-1.4-safe) ResFused pkl of the BEST-EMA snapshot (+ .bn.pkl) to --out.
Dropout intentionally NOT added (would change the fuseable deploy arch); regularization comes
from heavier aug + label smoothing + EMA + weight decay.

  CUDA_VISIBLE_DEVICES=0 python3 e11_train.py --seed 0 --steps 130000 --out ckpt/aug/aug_128x40_s0.pkl
"""
import os, sys, argparse, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn
import suit_aug, reflect_aug, dragon_aug

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, "data")


def smoothed_loss(logits, mask, y, eps):
    """Cross-entropy with label smoothing spread over the LEGAL action set of each sample.
    logits: (B,235) already has illegal ~ -1e38 (finite). We guard with torch.where so the
    illegal (-inf-ish) entries never enter the smoothing sum (avoids 0*inf=nan)."""
    logp = F.log_softmax(logits.float(), dim=1)
    nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)                      # (B,)
    legal = mask.float()
    logp_safe = torch.where(mask, logp, torch.zeros_like(logp))
    mean_legal = logp_safe.sum(1) / legal.sum(1).clamp(min=1.0)          # mean logp over legal
    loss = (1.0 - eps) * nll + eps * (-mean_legal)
    return loss.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--steps", type=int, default=130000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1.5e-4)
    ap.add_argument("--lsm", type=float, default=0.05)          # label smoothing
    ap.add_argument("--ema", type=float, default=0.999)         # EMA decay
    ap.add_argument("--p_suit", type=float, default=0.8)
    ap.add_argument("--p_ref", type=float, default=0.5)
    ap.add_argument("--p_drag", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valevery", type=int, default=4000)
    ap.add_argument("--teachers", default="ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(os.path.join(DDIR, "cooked_single.npz"))
    o, m, ac = d["obs"], d["mask"], d["act"].astype(np.int64)
    N = len(ac)
    # SAME split as e1_train (rng 12345) -> val_acc comparable to bn128s1
    rng = np.random.RandomState(12345); perm = rng.permutation(N)
    nval = min(50000, N // 20); vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    print(f"N={N:,} train={len(tidx):,} val={len(vidx):,} ch={a.channels} blk={a.blocks} "
          f"seed={a.seed} steps={a.steps} lsm={a.lsm} ema={a.ema} "
          f"p_suit/ref/drag={a.p_suit}/{a.p_ref}/{a.p_drag}", flush=True)

    # ----- GPU aug remap tensors -----
    S_rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in suit_aug.PERMS]
    S_Am = [torch.tensor(suit_aug.action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    S_Fm = [torch.tensor(suit_aug.fwd_action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    R_A = torch.tensor(reflect_aug.reflect_action(), device=dev, dtype=torch.long)
    R_F = torch.tensor(reflect_aug.fwd_reflect_action(), device=dev, dtype=torch.long)
    D_col = [torch.tensor(dragon_aug.obs_col_map(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Am = [torch.tensor(dragon_aug.action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Fm = [torch.tensor(dragon_aug.fwd_action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]

    net = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    from models_explore import build as _build
    teachers = []
    for tp in a.teachers.split(","):
        t = _build("resbn_fused", channels=128, blocks=40)
        t.load_state_dict(torch.load(tp, map_location="cpu")); t.eval(); t.to(dev)
        for p_ in t.parameters(): p_.requires_grad_(False)
        teachers.append(t)
    print(f"KD teachers: {len(teachers)} alpha={a.alpha}", flush=True)
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

    def lr_at(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        import math
        prog = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler()

    # EMA snapshot (float tensors EMA'd; int buffers copied)
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    is_float = {k: torch.is_floating_point(v) for k, v in net.state_dict().items()}

    def ema_update():
        sd = net.state_dict()
        for k, v in sd.items():
            if is_float[k]:
                ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
            else:
                ema[k] = v.detach().clone()

    ema_net = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    from models_explore import build as _build
    teachers = []
    for tp in a.teachers.split(","):
        t = _build("resbn_fused", channels=128, blocks=40)
        t.load_state_dict(torch.load(tp, map_location="cpu")); t.eval(); t.to(dev)
        for p_ in t.parameters(): p_.requires_grad_(False)
        teachers.append(t)
    print(f"KD teachers: {len(teachers)} alpha={a.alpha}", flush=True)

    def fetch(idx):
        idx = np.sort(idx)
        ob = torch.from_numpy(np.ascontiguousarray(o[idx])).to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(m[idx])).to(dev)            # bool
        y = torch.from_numpy(np.ascontiguousarray(ac[idx]).astype(np.int64)).to(dev)
        return ob, mk, y

    @torch.no_grad()
    def val_of(model):
        model.eval(); c = 0
        for i in range(0, len(vidx), 8192):
            b = vidx[i:i + 8192]
            ob, mk, y = fetch(b)
            pr = model({"is_training": False,
                        "obs": {"observation": ob, "action_mask": mk.float()}}).argmax(1)
            c += (pr == y).sum().item()
        return c / len(vidx)

    def eval_ema():
        ema_net.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        return val_of(ema_net)

    r2 = np.random.RandomState(1 + a.seed); best = 0.0; nt = len(tidx)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    bn_out = a.out[:-4] + ".bn.pkl"
    t0 = time.time(); net.train()
    for s in range(a.steps):
        b = tidx[r2.randint(0, nt, a.bs)]
        ob, mk, y = fetch(b)
        # compose verified augmentations (order irrelevant: disjoint tile groups / commute)
        if r2.random() < a.p_suit:
            pi = r2.randint(1, 6); ob = ob[:, :, S_rows[pi], :]; mk = mk[:, S_Am[pi]]; y = S_Fm[pi][y]
        if r2.random() < a.p_ref:
            ob2 = ob.clone(); ob2[:, :, :3, :] = ob[:, :, :3, :].flip(-1); ob = ob2
            mk = mk[:, R_A]; y = R_F[y]
        if r2.random() < a.p_drag:
            qi = r2.randint(1, 6); ob2 = ob.clone(); ob2[:, :, 3, :] = ob[:, :, 3, :][:, :, D_col[qi]]
            ob = ob2; mk = mk[:, D_Am[qi]]; y = D_Fm[qi][y]
        with torch.cuda.amp.autocast():
            logits = net({"is_training": True, "obs": {"observation": ob, "action_mask": mk.float()}})
            with torch.no_grad():
                tp_acc = None
                for t in teachers:
                    tl = t({"is_training": False, "obs": {"observation": ob, "action_mask": mk.float()}})
                    tl = tl.float(); tl = torch.where(mk, tl, torch.full_like(tl, -1e30))
                    p = torch.softmax(tl, 1)
                    tp_acc = p if tp_acc is None else tp_acc + p
                tsoft = tp_acc / len(teachers)                       # (B,235) mean softmax over legal
        logp = torch.nn.functional.log_softmax(torch.where(mk, logits.float(), torch.full_like(logits.float(), -1e30)), 1)
        kd = -(tsoft * logp).sum(1).mean()                           # CE toward ensemble soft targets
        loss = a.alpha * kd + (1.0 - a.alpha) * smoothed_loss(logits, mk, y, a.lsm)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ema_update()
        if s % a.valevery == 0 and s > 0:
            v = eval_ema()
            if v > best:
                best = v
                ema_net.load_state_dict({k: vv.to(dev) for k, vv in ema.items()})
                torch.save(ema_net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"emaval {v:.4f} best {best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    v = eval_ema()
    if v > best:
        best = v
        ema_net.load_state_dict({k: vv.to(dev) for k, vv in ema.items()})
        torch.save(ema_net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
    # fuse the BEST-EMA BN snapshot -> fused pkl for gating/deploy
    bestnet = ResBNCNN(channels=a.channels, blocks=a.blocks)
    bestnet.load_state_dict(torch.load(bn_out, map_location="cpu")); bestnet.eval()
    fused = fuse_resbn(bestnet)
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE best_ema_val={best:.4f} -> {a.out} (fused) + {bn_out} (bn)", flush=True)


if __name__ == "__main__":
    main()
