"""
e11plus_train.py — featplus (38+P planes) 128x40 BC trainer using the ENHANCED e11 recipe.

Motivation: the ladder's strongest bot is "ResNet18 + FEATURES". Prior featplus nets (featA/B/C)
were trained with the OLD plain-CE recipe (suit-aug only, no label-smoothing / EMA). This trainer
combines mechanism feature planes (danger/genbutsu-safe + shanten/useful-tile) with the PROVEN
enhanced recipe that aug_s0 uses (suit x reflect x dragon aug, label smoothing over legal actions,
EMA, warmup+cosine). This is the untested (features x enhanced-recipe) cell -> priority #1.

Feature planes (canonical order base,A,B,C  — MUST match featplus / SimPlus serve order):
  A (+5): opponent-river danger (per-opp discard-seen) + meld-commitment + game-progress
  B (+4): regular/7pairs/13orphans shanten (broadcast) + useful-tile plane  (from precomputed planeB.npy)
  C (+3): genbutsu safe-tile per opponent (furiten-safe discards)
All planes transform covariantly under suit-perm / rank-reflection / dragon-perm (genuine mahjong
symmetries): shanten & danger are invariant, spatial planes permute with the tiles -> aug stays
label-preserving. planeB.npy is index-aligned with cooked_single.npz (verified same ordering).

Same rng-12345 val split as e1/e11 -> val_acc directly comparable to aug_s0 (0.887).
Saves FUSED pkl (deploy) + .bn.pkl (unfused ResBNCNNP, used by parity_gate_plus).

  CUDA_VISIBLE_DEVICES=0 python3 e11plus_train.py --sets ABC --seed 0 --steps 130000 \
      --out ckpt/featx/featABC_e11_s0.pkl
"""
import os, sys, argparse, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_plus import ResBNCNNP
from models_explore import fuse_resbn
import suit_aug, reflect_aug, dragon_aug
import featplus

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, "data")


def smoothed_loss(logits, mask, y, eps):
    logp = F.log_softmax(logits.float(), dim=1)
    nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)
    legal = mask.float()
    logp_safe = torch.where(mask, logp, torch.zeros_like(logp))
    mean_legal = logp_safe.sum(1) / legal.sum(1).clamp(min=1.0)
    loss = (1.0 - eps) * nll + eps * (-mean_legal)
    return loss.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="ABC")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--steps", type=int, default=130000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1.5e-4)
    ap.add_argument("--lsm", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--p_suit", type=float, default=0.8)
    ap.add_argument("--p_ref", type=float, default=0.5)
    ap.add_argument("--p_drag", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valevery", type=int, default=5000)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--planeB", default=os.path.join(DDIR, "planeB.npy"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.dev
    sets = a.sets
    P = featplus.total_extra(sets)
    IN = 38 + P
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(os.path.join(DDIR, "cooked_single.npz"))
    o, m, ac = d["obs"], d["mask"], d["act"].astype(np.int64)
    N = len(ac)
    planeB = None
    if "B" in sets:
        planeB = np.load(a.planeB)  # full into RAM (~3.4GB); index-aligned with cooked_single
        assert planeB.shape[0] == N, f"planeB N {planeB.shape[0]} != {N}"
    rng = np.random.RandomState(12345); perm = rng.permutation(N)
    nval = min(50000, N // 20); vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    print(f"sets={sets} P={P} in_planes={IN}  N={N:,} train={len(tidx):,} val={len(vidx):,} "
          f"ch={a.channels} blk={a.blocks} seed={a.seed} steps={a.steps} lsm={a.lsm} ema={a.ema} "
          f"p_suit/ref/drag={a.p_suit}/{a.p_ref}/{a.p_drag} dev={dev}", flush=True)

    # GPU aug remap tensors
    S_rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in suit_aug.PERMS]
    S_Fm = [torch.tensor(suit_aug.fwd_action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    S_Am = [torch.tensor(suit_aug.action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    R_A = torch.tensor(reflect_aug.reflect_action(), device=dev, dtype=torch.long)
    R_F = torch.tensor(reflect_aug.fwd_reflect_action(), device=dev, dtype=torch.long)
    D_col = [torch.tensor(dragon_aug.obs_col_map(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Am = [torch.tensor(dragon_aug.action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Fm = [torch.tensor(dragon_aug.fwd_action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]

    net = ResBNCNNP(in_planes=IN, channels=a.channels, blocks=a.blocks).to(dev)
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

    def lr_at(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        prog = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    is_float = {k: torch.is_floating_point(v) for k, v in net.state_dict().items()}

    def ema_update():
        sd = net.state_dict()
        for k, v in sd.items():
            if is_float[k]:
                ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
            else:
                ema[k] = v.detach().clone()

    ema_net = ResBNCNNP(in_planes=IN, channels=a.channels, blocks=a.blocks).to(dev)

    def build_obs_np(idx):
        base = o[idx].astype(np.float32)          # (B,38,4,9)
        parts = [base]
        if "A" in sets: parts.append(featplus.planes_A(base))
        if "B" in sets: parts.append(planeB[idx].astype(np.float32))
        if "C" in sets: parts.append(featplus.planes_C(base))
        return np.concatenate(parts, axis=1)      # (B,IN,4,9)

    def fetch(idx):
        idx = np.sort(idx)
        ob = torch.from_numpy(np.ascontiguousarray(build_obs_np(idx))).to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(m[idx])).to(dev)   # bool
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
        if r2.random() < a.p_suit:
            pi = r2.randint(1, 6); ob = ob[:, :, S_rows[pi], :]; mk = mk[:, S_Am[pi]]; y = S_Fm[pi][y]
        if r2.random() < a.p_ref:
            ob2 = ob.clone(); ob2[:, :, :3, :] = ob[:, :, :3, :].flip(-1); ob = ob2
            mk = mk[:, R_A]; y = R_F[y]
        if r2.random() < a.p_drag:
            qi = r2.randint(1, 6); ob2 = ob.clone(); ob2[:, :, 3, :] = ob[:, :, 3, :][:, :, D_col[qi]]
            ob = ob2; mk = mk[:, D_Am[qi]]; y = D_Fm[qi][y]
        with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            logits = net({"is_training": True, "obs": {"observation": ob, "action_mask": mk.float()}})
        loss = smoothed_loss(logits, mk, y, a.lsm)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ema_update()
        if a.valevery > 0 and s % a.valevery == 0 and s > 0:
            v = eval_ema()
            if v > best:
                best = v
                ema_net.load_state_dict({k: vv.to(dev) for k, vv in ema.items()})
                torch.save(ema_net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"emaval {v:.4f} best {best:.4f} ({time.time()-t0:.0f}s)", flush=True)
            net.train()
    v = eval_ema()
    if v > best:
        best = v
        ema_net.load_state_dict({k: vv.to(dev) for k, vv in ema.items()})
        torch.save(ema_net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
    # fuse best-EMA -> fused pkl
    bestnet = ResBNCNNP(in_planes=IN, channels=a.channels, blocks=a.blocks)
    bestnet.load_state_dict(torch.load(bn_out, map_location="cpu")); bestnet.eval()
    fused = fuse_resbn(bestnet)
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE sets={sets} best_ema_val={best:.4f} -> {a.out} (fused) + {bn_out} (bn)", flush=True)


if __name__ == "__main__":
    main()
