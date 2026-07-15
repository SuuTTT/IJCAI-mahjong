"""e11_cond_train.py — SOURCE-CONDITIONED multi-corpus BC (2027 candidate, Track 3).

Hypothesis (from VALUE_V2: mixing sources w/o conditioning collapsed value r 0.71->0.36;
conditioning restored it): policy trained on base corpus (5.86M) + Final2 finalist corpus
(723k) with a SOURCE PLANE (39th input channel; 0=base, 1=final2) deployed with plane=1
should exceed both pure-corpus training (aug_s0 line) and unconditioned mixing
(F2_CORPUS_KD b_mix parity).

Arms: --plane src  (conditioned)   | --plane zero (unconditioned control, SAME 39-ch arch).
Recipe: identical to e11_train.py (suit/reflect/dragon aug, lsm, EMA, warmup+cosine).
Model selection on FINAL2 val acc (target distribution); base val logged.
"""
import os, sys, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import models_explore
models_explore.IN_PLANES = 39                      # BEFORE class instantiation
from models_explore import ResBNCNN, fuse_resbn
import suit_aug, reflect_aug, dragon_aug

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, "data")


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
    ap.add_argument("--f2_frac", type=float, default=0.5)   # per-batch final2 fraction
    ap.add_argument("--plane", choices=["src", "zero"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valevery", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(os.path.join(DDIR, "cooked_single.npz"))
    o, m, ac = d["obs"], d["mask"], d["act"].astype(np.int64)
    f = np.load("/root/final2_harvest/final2_cai_corpus.npz")
    fo, fm, fac = f["obs"], f["mask"], f["act"].astype(np.int64)
    N, FN = len(ac), len(fac)
    assert o.shape[1:] == (38, 4, 9) and fo.shape[1:] == (38, 4, 9), "obs layout mismatch"

    rng = np.random.RandomState(12345); perm = rng.permutation(N)
    nval = min(50000, N // 20); vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    frng = np.random.RandomState(54321); fperm = frng.permutation(FN)
    fnval = min(50000, FN // 10); fvidx = np.sort(fperm[:fnval]); ftidx = fperm[fnval:]
    print(f"base N={N:,} f2 N={FN:,} plane={a.plane} f2_frac={a.f2_frac} seed={a.seed} "
          f"steps={a.steps} ch={a.channels} blk={a.blocks}", flush=True)

    S_rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in suit_aug.PERMS]
    S_Am = [torch.tensor(suit_aug.action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    S_Fm = [torch.tensor(suit_aug.fwd_action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    R_A = torch.tensor(reflect_aug.reflect_action(), device=dev, dtype=torch.long)
    R_F = torch.tensor(reflect_aug.fwd_reflect_action(), device=dev, dtype=torch.long)
    D_col = [torch.tensor(dragon_aug.obs_col_map(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Am = [torch.tensor(dragon_aug.action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Fm = [torch.tensor(dragon_aug.fwd_action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]

    net = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    assert net.stem[0].weight.shape[1] == 39, "conditioning plane not in stem"
    print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

    def lr_at(step):
        if step < a.warmup: return (step + 1) / a.warmup
        import math; prog = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler()
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    is_float = {k: torch.is_floating_point(v) for k, v in net.state_dict().items()}

    def ema_update():
        for k, v in net.state_dict().items():
            if is_float[k]: ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
            else: ema[k] = v.detach().clone()
    ema_net = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)

    def fetch(src_arrs, idx, src_val):
        oo, mm, aa = src_arrs
        idx = np.sort(idx)
        ob = torch.from_numpy(np.ascontiguousarray(oo[idx])).to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(mm[idx])).to(dev)
        y = torch.from_numpy(np.ascontiguousarray(aa[idx]).astype(np.int64)).to(dev)
        sv = src_val if a.plane == "src" else 0.0
        pl = torch.full((len(idx), 1, 4, 9), sv, device=dev, dtype=ob.dtype)
        return ob, mk, y, pl

    @torch.no_grad()
    def val_of(model, arrs, idxs, src_val):
        model.eval(); c = 0
        for i in range(0, len(idxs), 8192):
            ob, mk, y, pl = fetch(arrs, idxs[i:i + 8192], src_val)
            ob = torch.cat([ob.float(), pl.float()], 1)
            pr = model({"is_training": False,
                        "obs": {"observation": ob, "action_mask": mk.float()}}).argmax(1)
            c += (pr == y).sum().item()
        return c / len(idxs)

    def eval_ema():
        ema_net.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        return (val_of(ema_net, (fo, fm, fac), fvidx, 1.0),
                val_of(ema_net, (o, m, ac), vidx, 0.0))

    r2 = np.random.RandomState(1 + a.seed); best = 0.0
    nb = int(round(a.bs * (1 - a.f2_frac))); nf = a.bs - nb
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    bn_out = a.out[:-4] + ".bn.pkl"; hist = []
    t0 = time.time(); net.train()
    for s in range(a.steps):
        ob1, mk1, y1, pl1 = fetch((o, m, ac), tidx[r2.randint(0, len(tidx), nb)], 0.0)
        ob2, mk2, y2, pl2 = fetch((fo, fm, fac), ftidx[r2.randint(0, len(ftidx), nf)], 1.0)
        ob = torch.cat([ob1, ob2]); mk = torch.cat([mk1, mk2])
        y = torch.cat([y1, y2]); pl = torch.cat([pl1, pl2])
        if r2.random() < a.p_suit:
            pi = r2.randint(1, 6); ob = ob[:, :, S_rows[pi], :]; pl = pl[:, :, S_rows[pi], :]
            mk = mk[:, S_Am[pi]]; y = S_Fm[pi][y]
        if r2.random() < a.p_ref:
            ob2_ = ob.clone(); ob2_[:, :, :3, :] = ob[:, :, :3, :].flip(-1); ob = ob2_
            mk = mk[:, R_A]; y = R_F[y]
        if r2.random() < a.p_drag:
            qi = r2.randint(1, 6); ob2_ = ob.clone(); ob2_[:, :, 3, :] = ob[:, :, 3, :][:, :, D_col[qi]]
            ob = ob2_; mk = mk[:, D_Am[qi]]; y = D_Fm[qi][y]
        ob = torch.cat([ob.float(), pl.float()], 1)  # plane constant -> aug-invariant anyway
        with torch.cuda.amp.autocast():
            logits = net({"is_training": True, "obs": {"observation": ob, "action_mask": mk.float()}})
        loss = smoothed_loss(logits, mk, y, a.lsm)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ema_update()
        if s % a.valevery == 0 and s > 0:
            vf2, vb = eval_ema()
            hist.append(dict(step=s, loss=round(loss.item(), 4), val_f2=round(vf2, 4), val_base=round(vb, 4)))
            if vf2 > best:   # selection on TARGET (final2) distribution
                best = vf2
                torch.save(ema_net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} valF2 {vf2:.4f} valBase {vb:.4f} "
                  f"best {best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    vf2, vb = eval_ema()
    hist.append(dict(step=a.steps, val_f2=round(vf2, 4), val_base=round(vb, 4)))
    if vf2 > best:
        best = vf2; torch.save(ema_net.state_dict(), bn_out, _use_new_zipfile_serialization=False)
    bestnet = ResBNCNN(channels=a.channels, blocks=a.blocks)
    bestnet.load_state_dict(torch.load(bn_out, map_location="cpu")); bestnet.eval()
    fused = fuse_resbn(bestnet)
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    json.dump(dict(plane=a.plane, seed=a.seed, f2_frac=a.f2_frac, steps=a.steps,
                   best_val_f2=round(best, 4), hist=hist),
              open(a.out + ".traininfo.json", "w"), indent=1)
    print(f"DONE plane={a.plane} seed={a.seed} best_val_f2={best:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
