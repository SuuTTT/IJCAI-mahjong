"""
e13_kd_corpus.py — FINALIST-CORPUS DISTILLATION trainer (Exp1 of the Final2 campaign).

Fork of e13_kd_train.py (identical KD recipe: 6-aug-teacher KD on cooked_single, suit/
reflect/dragon aug, label smoothing, EMA, cosine schedule, fused save) plus a
behavior-cloning CE term on the Final2 finalist corpus (cai encoding):

    mixed (default): loss = alpha*KD(T6) + (1-alpha)*smoothedCE(cooked) + beta*smoothedCE(corpus)
    --pure         : loss = smoothedCE(corpus)   (no KD, no cooked; reference arm)

Corpus rows are filtered to bots in --bots (comma ids; kong=0 moyu=1 QiuQiuR=2 player152=3)
and to non-forced decisions (mask.sum()>1). Corpus val split is BY GAME (rng 4242, 5% games).
The same on-GPU augmentations are applied to corpus batches (label-preserving, verified).

Sidecar <out>.json: cooked val acc (comparable to kdens 0.8836 line), corpus val acc, times.
"""
import os, sys, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import ResBNCNN, fuse_resbn, build as _build
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
    ap.add_argument("--bs_c", type=int, default=512)
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
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--bots", default="")            # "" = all 4; "0,1" = kong+moyu
    ap.add_argument("--pure", action="store_true")   # corpus-only BC
    ap.add_argument("--corpus", default="/root/final2_harvest/final2_cai_corpus.npz")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    # ---- corpus ----
    dc = np.load(a.corpus)
    co, cm, ca = dc["obs"], dc["mask"], dc["act"].astype(np.int64)
    cbot, cgame = dc["bot"], dc["game"]
    keep = cm.sum(1) > 1
    if a.bots:
        want = np.array([int(x) for x in a.bots.split(",")])
        keep &= np.isin(cbot, want)
    co, cm, ca, cgame = co[keep], cm[keep], ca[keep], cgame[keep]
    ug = np.unique(cgame)
    rngc = np.random.RandomState(4242); permg = rngc.permutation(len(ug))
    val_games = set(ug[permg[:max(1, len(ug) // 20)]].tolist())
    is_val = np.isin(cgame, list(val_games))
    cvidx = np.flatnonzero(is_val); ctidx = np.flatnonzero(~is_val)
    if len(cvidx) > 60000:
        cvidx = cvidx[np.random.RandomState(9).permutation(len(cvidx))[:60000]]
        cvidx = np.sort(cvidx)
    print(f"corpus N={len(ca):,} (bots={a.bots or 'all'}) train={len(ctidx):,} "
          f"val={len(cvidx):,} ({len(val_games)} games) pure={a.pure} beta={a.beta}", flush=True)

    # ---- cooked (skip full load in pure mode except for val) ----
    d = np.load(a.data) if a.data else np.load(os.path.join(DDIR, "cooked_single.npz"))
    o, m, ac = d["obs"], d["mask"], d["act"].astype(np.int64)
    N = len(ac)
    rng = np.random.RandomState(12345); perm = rng.permutation(N)
    nval = min(50000, N // 20); vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    print(f"cooked N={N:,} train={len(tidx):,} val={len(vidx):,} seed={a.seed} steps={a.steps}",
          flush=True)

    S_rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in suit_aug.PERMS]
    S_Am = [torch.tensor(suit_aug.action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    S_Fm = [torch.tensor(suit_aug.fwd_action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    R_A = torch.tensor(reflect_aug.reflect_action(), device=dev, dtype=torch.long)
    R_F = torch.tensor(reflect_aug.fwd_reflect_action(), device=dev, dtype=torch.long)
    D_col = [torch.tensor(dragon_aug.obs_col_map(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Am = [torch.tensor(dragon_aug.action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Fm = [torch.tensor(dragon_aug.fwd_action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]

    def augment(r2, ob, mk, y):
        if r2.random() < a.p_suit:
            pi = r2.randint(1, 6); ob = ob[:, :, S_rows[pi], :]; mk = mk[:, S_Am[pi]]; y = S_Fm[pi][y]
        if r2.random() < a.p_ref:
            ob2 = ob.clone(); ob2[:, :, :3, :] = ob[:, :, :3, :].flip(-1); ob = ob2
            mk = mk[:, R_A]; y = R_F[y]
        if r2.random() < a.p_drag:
            qi = r2.randint(1, 6); ob2 = ob.clone(); ob2[:, :, 3, :] = ob[:, :, 3, :][:, :, D_col[qi]]
            ob = ob2; mk = mk[:, D_Am[qi]]; y = D_Fm[qi][y]
        return ob, mk, y

    net = ResBNCNN(channels=a.channels, blocks=a.blocks).to(dev)
    teachers = []
    if not a.pure:
        for tp in a.teachers.split(","):
            t = _build("resbn_fused", channels=128, blocks=40)
            t.load_state_dict(torch.load(tp, map_location="cpu")); t.eval(); t.to(dev)
            for p_ in t.parameters():
                p_.requires_grad_(False)
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

    def fetch_from(oA, mA, aA, idx):
        idx = np.sort(idx)
        ob = torch.from_numpy(np.ascontiguousarray(oA[idx])).to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(mA[idx])).to(dev)
        y = torch.from_numpy(np.ascontiguousarray(aA[idx]).astype(np.int64)).to(dev)
        return ob, mk, y

    @torch.no_grad()
    def acc_of(model, oA, mA, aA, idxs):
        model.eval(); c = 0
        for i in range(0, len(idxs), 8192):
            b = idxs[i:i + 8192]
            ob, mk, y = fetch_from(oA, mA, aA, b)
            pr = model({"is_training": False,
                        "obs": {"observation": ob, "action_mask": mk.float()}}).argmax(1)
            c += (pr == y).sum().item()
        return c / max(1, len(idxs))

    def eval_ema():
        ema_net.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        if a.pure:
            return acc_of(ema_net, co, cm, ca, cvidx)
        return acc_of(ema_net, o, m, ac, vidx)

    r2 = np.random.RandomState(1 + a.seed)
    r3 = np.random.RandomState(101 + a.seed)
    best = 0.0; nt = len(tidx); ntc = len(ctidx)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    bn_out = a.out[:-4] + ".bn.pkl"
    t0 = time.time(); net.train()
    for s in range(a.steps):
        with torch.cuda.amp.autocast():
            if a.pure:
                bc = ctidx[r3.randint(0, ntc, a.bs)]
                obc, mkc, yc = fetch_from(co, cm, ca, bc)
                obc, mkc, yc = augment(r3, obc, mkc, yc)
                logits_c = net({"is_training": True,
                                "obs": {"observation": obc, "action_mask": mkc.float()}})
                loss = smoothed_loss(logits_c, mkc, yc, a.lsm)
            else:
                b = tidx[r2.randint(0, nt, a.bs)]
                ob, mk, y = fetch_from(o, m, ac, b)
                ob, mk, y = augment(r2, ob, mk, y)
                logits = net({"is_training": True,
                              "obs": {"observation": ob, "action_mask": mk.float()}})
                with torch.no_grad():
                    tp_acc = None
                    for t in teachers:
                        tl = t({"is_training": False,
                                "obs": {"observation": ob, "action_mask": mk.float()}})
                        tl = tl.float(); tl = torch.where(mk, tl, torch.full_like(tl, -1e30))
                        p = torch.softmax(tl, 1)
                        tp_acc = p if tp_acc is None else tp_acc + p
                    tsoft = tp_acc / len(teachers)
                logp = F.log_softmax(torch.where(mk, logits.float(),
                                                 torch.full_like(logits.float(), -1e30)), 1)
                kd = -(tsoft * logp).sum(1).mean()
                loss = a.alpha * kd + (1.0 - a.alpha) * smoothed_loss(logits, mk, y, a.lsm)
                if a.beta > 0:
                    bc = ctidx[r3.randint(0, ntc, a.bs_c)]
                    obc, mkc, yc = fetch_from(co, cm, ca, bc)
                    obc, mkc, yc = augment(r3, obc, mkc, yc)
                    logits_c = net({"is_training": True,
                                    "obs": {"observation": obc, "action_mask": mkc.float()}})
                    loss = loss + a.beta * smoothed_loss(logits_c, mkc, yc, a.lsm)
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
    bestnet = ResBNCNN(channels=a.channels, blocks=a.blocks)
    bestnet.load_state_dict(torch.load(bn_out, map_location="cpu")); bestnet.eval()
    fused = fuse_resbn(bestnet)
    torch.save(fused.state_dict(), a.out, _use_new_zipfile_serialization=False)
    bestnet.to(dev)
    cooked_val = acc_of(bestnet, o, m, ac, vidx)
    corpus_val = acc_of(bestnet, co, cm, ca, cvidx)
    with open(a.out[:-4] + ".json", "w") as f:
        json.dump(dict(arm=("pure" if a.pure else f"mix_beta{a.beta}_bots{a.bots or 'all'}"),
                       seed=a.seed, steps=a.steps, best_ema_val=round(best, 4),
                       cooked_val_acc=round(float(cooked_val), 4),
                       corpus_val_acc=round(float(corpus_val), 4),
                       corpus_rows_train=int(len(ctidx)),
                       seconds=round(time.time() - t0, 1)), f)
    print(f"DONE seed{a.seed} best={best:.4f} cooked_val={cooked_val:.4f} "
          f"corpus_val={corpus_val:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
