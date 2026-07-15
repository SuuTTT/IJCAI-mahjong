"""
seq_bc.py — BC trainer for the TEMPORAL CNN+GRU net (obs + ordered discard sequence).
Uses cooked_single.npz (obs/mask/act) + cooked_seq.npz (seq/act, verified aligned). Enhanced
recipe: suit x reflect x dragon aug applied CONSISTENTLY to obs AND seq (token relabel LUTs
derived from the same action perms), label smoothing over legal actions, EMA, warmup+cosine.
Same val split (rng 12345) as e11 -> val comparable to aug_s0 (0.887). Saves raw best-EMA state_dict.
--kind {temporal, cnnonly}. cnnonly = CNN-branch control (GRU signal zeroed), for the ablation.
"""
import os, sys, argparse, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_seq import build_seq, PAD, VOCAB
import suit_aug, reflect_aug, dragon_aug

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, "data")


def smoothed_loss(logits, mask, y, eps):
    logp = F.log_softmax(logits.float(), dim=1)
    nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)
    logp_safe = torch.where(mask, logp, torch.zeros_like(logp))
    mean_legal = logp_safe.sum(1) / mask.float().sum(1).clamp(min=1.0)
    return ((1.0 - eps) * nll + eps * (-mean_legal)).mean()


def tile_perm_from_fwd(fwd):
    # fwd: (235,) old_action -> new_action. Play block = idx 2..35. tile_perm[t] = fwd[t+2]-2
    return np.array([int(fwd[t + 2]) - 2 for t in range(34)], dtype=np.int64)


def tok_lut(tile_perm):
    lut = np.arange(VOCAB, dtype=np.int64)  # includes PAD=136 -> stays
    for tok in range(136):
        tile, rel = tok // 4, tok % 4
        lut[tok] = tile_perm[tile] * 4 + rel
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="temporal")
    ap.add_argument("--channels", type=int, default=128); ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--emb", type=int, default=64); ap.add_argument("--gru", type=int, default=256)
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--tf_layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100000); ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1.5e-4); ap.add_argument("--lsm", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--p_suit", type=float, default=0.8); ap.add_argument("--p_ref", type=float, default=0.5)
    ap.add_argument("--p_drag", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--valevery", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"; torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(os.path.join(DDIR, "cooked_single.npz"))
    o, m, ac = d["obs"], d["mask"], d["act"].astype(np.int64)
    ds = np.load(os.path.join(DDIR, "cooked_seq.npz"))
    seq = ds["seq"].astype(np.int64); sact = ds["act"].astype(np.int64)
    assert len(seq) == len(ac) and np.array_equal(sact, ac.astype(np.int64)), "SEQ/OBS misaligned!"
    N = len(ac)
    rng = np.random.RandomState(12345); perm = rng.permutation(N)
    nval = min(50000, N // 20); vidx = np.sort(perm[:nval]); tidx = perm[nval:]
    print(f"N={N:,} train={len(tidx):,} val={len(vidx):,} kind={a.kind} ch={a.channels} blk={a.blocks} "
          f"emb={a.emb} gru={a.gru} seed={a.seed} steps={a.steps}", flush=True)

    # ---- aug remaps (obs) + matching token LUTs (seq) ----
    S_rows = [torch.tensor([p[0], p[1], p[2], 3], device=dev) for p in suit_aug.PERMS]
    S_Am = [torch.tensor(suit_aug.action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    S_Fm = [torch.tensor(suit_aug.fwd_action_perm(p), device=dev, dtype=torch.long) for p in suit_aug.PERMS]
    S_tok = [torch.tensor(tok_lut(tile_perm_from_fwd(suit_aug.fwd_action_perm(p))), device=dev) for p in suit_aug.PERMS]
    R_A = torch.tensor(reflect_aug.reflect_action(), device=dev, dtype=torch.long)
    R_F = torch.tensor(reflect_aug.fwd_reflect_action(), device=dev, dtype=torch.long)
    R_tok = torch.tensor(tok_lut(tile_perm_from_fwd(reflect_aug.fwd_reflect_action())), device=dev)
    D_col = [torch.tensor(dragon_aug.obs_col_map(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Am = [torch.tensor(dragon_aug.action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_Fm = [torch.tensor(dragon_aug.fwd_action_perm(q), device=dev, dtype=torch.long) for q in dragon_aug.PERMS_D]
    D_tok = [torch.tensor(tok_lut(tile_perm_from_fwd(dragon_aug.fwd_action_perm(q))), device=dev) for q in dragon_aug.PERMS_D]

    _scfg = dict(channels=a.channels, blocks=a.blocks, emb=a.emb, gru=a.gru, gru_layers=a.gru_layers, heads=a.heads, tf_layers=a.tf_layers)
    net = build_seq(a.kind, **_scfg).to(dev)
    nparams = sum(p.numel() for p in net.parameters()); print(f"params {nparams:,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

    def lr_at(step):
        if step < a.warmup: return (step + 1) / a.warmup
        prog = (step - a.warmup) / max(1, a.steps - a.warmup); return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at); scaler = torch.cuda.amp.GradScaler()

    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    is_float = {k: torch.is_floating_point(v) for k, v in net.state_dict().items()}
    def ema_update():
        for k, v in net.state_dict().items():
            if is_float[k]: ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1 - a.ema)
            else: ema[k] = v.detach().clone()
    ema_net = build_seq(a.kind, **_scfg).to(dev)

    def fetch(idx):
        idx = np.sort(idx)
        ob = torch.from_numpy(np.ascontiguousarray(o[idx])).to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(m[idx])).to(dev)
        sq = torch.from_numpy(np.ascontiguousarray(seq[idx])).to(dev)
        y = torch.from_numpy(np.ascontiguousarray(ac[idx])).to(dev)
        return ob, mk, sq, y

    @torch.no_grad()
    def val_of(model):
        model.eval(); c = 0
        for i in range(0, len(vidx), 8192):
            b = vidx[i:i + 8192]; ob, mk, sq, y = fetch(b)
            pr = model({"is_training": False, "seq": sq,
                        "obs": {"observation": ob, "action_mask": mk.float()}}).argmax(1)
            c += (pr == y).sum().item()
        return c / len(vidx)
    def eval_ema():
        ema_net.load_state_dict({k: v.to(dev) for k, v in ema.items()}); return val_of(ema_net)

    r2 = np.random.RandomState(1 + a.seed); best = 0.0; nt = len(tidx)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); t0 = time.time(); net.train()
    for s in range(a.steps):
        b = tidx[r2.randint(0, nt, a.bs)]; ob, mk, sq, y = fetch(b)
        if r2.random() < a.p_suit:
            pi = r2.randint(1, 6); ob = ob[:, :, S_rows[pi], :]; mk = mk[:, S_Am[pi]]; y = S_Fm[pi][y]; sq = S_tok[pi][sq]
        if r2.random() < a.p_ref:
            ob2 = ob.clone(); ob2[:, :, :3, :] = ob[:, :, :3, :].flip(-1); ob = ob2
            mk = mk[:, R_A]; y = R_F[y]; sq = R_tok[sq]
        if r2.random() < a.p_drag:
            qi = r2.randint(1, 6); ob2 = ob.clone(); ob2[:, :, 3, :] = ob[:, :, 3, :][:, :, D_col[qi]]
            ob = ob2; mk = mk[:, D_Am[qi]]; y = D_Fm[qi][y]; sq = D_tok[qi][sq]
        with torch.cuda.amp.autocast():
            logits = net({"is_training": True, "seq": sq,
                          "obs": {"observation": ob, "action_mask": mk.float()}})
        loss = smoothed_loss(logits, mk, y, a.lsm)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ema_update()
        if s % a.valevery == 0 and s > 0:
            v = eval_ema()
            if v > best:
                best = v; ema_net.load_state_dict({k: vv.to(dev) for k, vv in ema.items()})
                torch.save(ema_net.state_dict(), a.out, _use_new_zipfile_serialization=False)
            print(f"  step {s}/{a.steps} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"emaval {v:.4f} best {best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    v = eval_ema()
    if v > best:
        best = v; ema_net.load_state_dict({k: vv.to(dev) for k, vv in ema.items()})
        torch.save(ema_net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE kind={a.kind} best_ema_val={best:.4f} params={nparams} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
