"""
e13_kd_danger.py — DANGER-PENALIZED JOINT-DEFENSE KD (fork of e13_kd_train.py).

Tests whether defense LEARNED JOINTLY IN TRAINING (not bolted on at inference) can cut deal-ins
without the offense cost seen with the inference-time fold.  Loss:

    loss = alpha * KD(ensemble soft targets) + (1-alpha) * smoothed_CE + lam_danger * pen
    pen  = E_batch[ sum_{discard a} p_student(a|s) * sigma(danger_head(s)[a]) ]

The danger head (danger4.bn.pkl, frozen ResBNCNN 128x40, outputs per-discard deal-in logits at
action slots 2..35) is scored on the SAME (augmented) obs as the student each step, so the penalty
is exact under suit/reflect/dragon augmentation and leak-free (no label information — the head was
trained on deal-in outcomes, not on the BC action).  lam_danger=0 is the plain-KD control (penalty
still computed for logging, never added).

Everything else (data, split, augs, label smoothing, EMA, schedule, fused save) identical to
e13_kd_train.py so lam is the ONLY manipulated variable.  Also logs mean chosen-discard danger of
the best-EMA snapshot on the (un-augmented) val set -> <out>.json.
"""
import os, sys, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn, build as _build
import suit_aug, reflect_aug, dragon_aug

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, "data")
PLAY0, NPLAY = 2, 34


def smoothed_loss(logits, mask, y, eps):
    logp = F.log_softmax(logits.float(), dim=1)
    nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)
    legal = mask.float()
    logp_safe = torch.where(mask, logp, torch.zeros_like(logp))
    mean_legal = logp_safe.sum(1) / legal.sum(1).clamp(min=1.0)
    return ((1.0 - eps) * nll + eps * (-mean_legal)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60000)
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
    ap.add_argument("--valevery", type=int, default=4000)
    ap.add_argument("--teachers", default="ckpt/aug/aug_128x40_s0.pkl,ckpt/aug/aug_128x40_s1.pkl,ckpt/aug/aug_128x40_s2.pkl,ckpt/aug/aug_128x40_s3.pkl,ckpt/aug/aug_128x40_s4.pkl,ckpt/aug/aug_128x40_s5.pkl")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--lam_danger", type=float, required=True)
    ap.add_argument("--danger", default="ckpt/danger/danger4.bn.pkl")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(a.data) if a.data else np.load(os.path.join(DDIR, "cooked_single.npz"))
    o, m, ac = d["obs"], d["mask"], d["act"].astype(np.int64)
    N = len(ac)
    rng = np.random.RandomState(12345); perm = rng.permutation(N)
    nval = min(50000, N // 20); vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    print(f"N={N:,} train={len(tidx):,} val={len(vidx):,} lam_danger={a.lam_danger} seed={a.seed} "
          f"steps={a.steps}", flush=True)

    S_rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in suit_aug.PERMS]
    S_Am = [torch.tensor(suit_aug.action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    S_Fm = [torch.tensor(suit_aug.fwd_action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    R_A = torch.tensor(reflect_aug.reflect_action(), device=dev, dtype=torch.long)
    R_F = torch.tensor(reflect_aug.fwd_reflect_action(), device=dev, dtype=torch.long)
    D_col = [torch.tensor(dragon_aug.obs_col_map(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Am = [torch.tensor(dragon_aug.action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Fm = [torch.tensor(dragon_aug.fwd_action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]

    net = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    teachers = []
    for tp in a.teachers.split(","):
        t = _build("resbn_fused", channels=128, blocks=40)
        t.load_state_dict(torch.load(tp, map_location="cpu")); t.eval(); t.to(dev)
        for p_ in t.parameters():
            p_.requires_grad_(False)
        teachers.append(t)
    danger = ResBNCNN(channels=128, blocks=40)
    danger.load_state_dict(torch.load(a.danger, map_location="cpu")); danger.eval().to(dev)
    for p_ in danger.parameters():
        p_.requires_grad_(False)
    print(f"KD teachers: {len(teachers)} alpha={a.alpha}; danger head loaded ({a.danger})", flush=True)
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

    def fetch(idx):
        idx = np.sort(idx)
        ob = torch.from_numpy(np.ascontiguousarray(o[idx])).to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(m[idx])).to(dev)
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

    @torch.no_grad()
    def chosen_danger_of(model):
        """mean deal-in prob of the model's argmax DISCARD on the un-augmented val set."""
        model.eval(); dsum = 0.0; ndec = 0
        for i in range(0, len(vidx), 8192):
            b = vidx[i:i + 8192]
            ob, mk, _ = fetch(b)
            pr = model({"is_training": False,
                        "obs": {"observation": ob, "action_mask": mk.float()}}).argmax(1)
            dgl = danger({"is_training": False,
                          "obs": {"observation": ob, "action_mask": torch.ones(ob.shape[0], 235, device=dev)}})
            dg34 = torch.sigmoid(dgl[:, PLAY0:PLAY0 + NPLAY].float())
            play = pr - PLAY0; isplay = (play >= 0) & (play < NPLAY)
            if isplay.any():
                dsum += dg34[isplay, play[isplay]].sum().item(); ndec += int(isplay.sum().item())
        return dsum / max(ndec, 1)

    r2 = np.random.RandomState(1 + a.seed); best = 0.0; nt = len(tidx)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    bn_out = a.out[:-4] + ".bn.pkl"
    pen_run = None
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
        with torch.cuda.amp.autocast():
            logits = net({"is_training": True, "obs": {"observation": ob, "action_mask": mk.float()}})
            with torch.no_grad():
                tp_acc = None
                for t in teachers:
                    tl = t({"is_training": False, "obs": {"observation": ob, "action_mask": mk.float()}})
                    tl = tl.float(); tl = torch.where(mk, tl, torch.full_like(tl, -1e30))
                    p = torch.softmax(tl, 1)
                    tp_acc = p if tp_acc is None else tp_acc + p
                tsoft = tp_acc / len(teachers)
                dgl = danger({"is_training": False,
                              "obs": {"observation": ob, "action_mask": torch.ones(ob.shape[0], 235, device=dev)}})
                dg34 = torch.sigmoid(dgl[:, PLAY0:PLAY0 + NPLAY].float())        # (B,34) frozen
        logits_f = torch.where(mk, logits.float(), torch.full_like(logits.float(), -1e30))
        logp = F.log_softmax(logits_f, 1)
        kd = -(tsoft * logp).sum(1).mean()
        p_student = torch.softmax(logits_f, 1)
        pen = (p_student[:, PLAY0:PLAY0 + NPLAY] * dg34).sum(1).mean()            # expected deal-in
        pen_run = pen.item() if pen_run is None else 0.99 * pen_run + 0.01 * pen.item()
        loss = a.alpha * kd + (1.0 - a.alpha) * smoothed_loss(logits, mk, y, a.lsm)
        if a.lam_danger > 0:
            loss = loss + a.lam_danger * pen
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ema_update()
        if s % a.valevery == 0 and s > 0:
            v = eval_ema()
            if v > best:
                best = v
                ema_net.load_state_dict({k: vv.to(dev) for k, vv in ema.items()})
                torch.save(ema_net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} pen(ema) {pen_run:.4f} "
                  f"emaval {v:.4f} best {best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    v = eval_ema()
    if v > best:
        best = v
        ema_net.load_state_dict({k: vv.to(dev) for k, vv in ema.items()})
        torch.save(ema_net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
    bestnet = ResBNCNN(channels=a.channels, blocks=a.blocks)
    bestnet.load_state_dict(torch.load(bn_out, map_location="cpu")); bestnet.eval()
    fused = fuse_resbn(bestnet)
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    bestnet.to(dev)
    cd = chosen_danger_of(bestnet)
    with open(a.out[:-4] + ".json", "w") as f:
        json.dump(dict(lam_danger=a.lam_danger, seed=a.seed, steps=a.steps, best_ema_val=round(best, 4),
                       val_chosen_danger=round(float(cd), 4), pen_train_ema=round(float(pen_run), 4),
                       seconds=round(time.time() - t0, 1)), f)
    print(f"DONE lam={a.lam_danger} s{a.seed} best_ema_val={best:.4f} chosen_danger={cd:.4f} "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
