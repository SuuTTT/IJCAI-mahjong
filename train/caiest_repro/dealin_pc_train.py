"""
dealin_pc_train.py -- PER-CANDIDATE deal-in predictor: P(deal-in | state, candidate tile T).

Unlike dealin_train.py (which predicts P(deal-in|state) marginalizing over the chosen tile),
this conditions on WHICH tile is discarded by adding a 39th one-hot "candidate tile" plane to
the 38-plane cai obs. The plane is 1 at grid cell ((T-2)//9, (T-2)%9) -- the SAME (4,9) layout
(OFFSET_TILE, reshape (,4,9)) used by the HAND/DISCARD planes, so the conv sees the candidate
co-located with the hand. Trunk = ResBN 128x40, 1-logit head, plain BCE, EMA, fused deploy ckpt.

Trained on observed (state, chosen_tile, dealt_in) rows: label 1 iff that chosen discard was the
Ron tile (same validated label derivation as dealin_train.py). At deploy, swap the candidate plane
per legal tile T to rank each discard by P(Ron).

CAVEAT (measured + reported): the training distribution of T is kdens3's chosen discards, not
uniform over legal tiles -> possibly miscalibrated for tiles kdens3 rarely discards. Diagnostic:
val AUROC restricted to rows with >1 legal discard (the decisions that actually matter for ranking).

  CUDA_VISIBLE_DEVICES=0 python3 dealin_pc_train.py --seed 0 --steps 50000 --out ckpt/dealin_pc/dealin_pc_s0.pt
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

IN_PLANES, GRID = 39, 4 * 9          # 38 cai planes + 1 candidate-tile plane
PLAY, CHI = 2, 36
CORPUS = "/root/final2_harvest/final2_cai_corpus.npz"

class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return torch.relu(x + y)

class DealInNet(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU(),
                                  nn.Linear(512, 1))
    def forward(self, x):
        return self.foot(self.body(self.stem(x))).squeeze(-1)

class _FusedBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True); self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
    def forward(self, x):
        y = torch.relu(self.c1(x)); y = self.c2(y); return torch.relu(x + y)

class DealInFused(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__()
        self.stem = nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*(_FusedBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU(),
                                  nn.Linear(512, 1))
    def forward(self, x):
        return self.foot(self.body(torch.relu(self.stem(x)))).squeeze(-1)

def fuse_dealin(net):
    net.eval(); ch = net.stem[0].out_channels; blocks = len(net.body)
    f = DealInFused(ch, blocks).eval()
    f.stem.load_state_dict(fuse_conv_bn_eval(net.stem[0], net.stem[1]).state_dict())
    for i, blk in enumerate(net.body):
        f.body[i].c1.load_state_dict(fuse_conv_bn_eval(blk.c1, blk.b1).state_dict())
        f.body[i].c2.load_state_dict(fuse_conv_bn_eval(blk.c2, blk.b2).state_dict())
    f.foot.load_state_dict(net.foot.state_dict())
    return f

def derive_labels(game, seat, act, score, fan, step):
    N = len(act); labels = np.zeros(N, np.int8)
    order = np.argsort(game, kind='stable'); gs = game[order]
    ug, first = np.unique(gs, return_index=True); bnd = np.append(first, N)
    ron = zimo = draw = 0
    for gi in range(len(ug)):
        rows = order[bnd[gi]:bnd[gi + 1]]
        ss = seat[rows]; sc = score[rows]; fa = fan[rows]; st = step[rows]; aa = act[rows]
        seats = np.unique(ss)
        sscore = {s: sc[ss == s][0] for s in seats}; sfan = {s: fa[ss == s].max() for s in seats}
        winners = [s for s in seats if sfan[s] > 0]
        if not winners:
            draw += 1; continue
        w = winners[0]; losers = {s: sscore[s] for s in seats if s != w}
        if not losers: continue
        mn = min(losers.values()); mins = [s for s, v in losers.items() if v == mn]
        if len(mins) == 1 and len(set(losers.values())) > 1:
            ron += 1; disc = mins[0]; dm = (ss == disc) & (aa >= PLAY) & (aa < CHI)
            if dm.any(): dr = rows[dm]; labels[dr[np.argmax(st[dm])]] = 1
        else:
            zimo += 1
    return labels, dict(ron=ron, zimo=zimo, draw=draw, games=len(ug))

def calib_deciles(y, p, nb=10):
    idx = np.argsort(p); y = y[idx]; p = p[idx]; out = []
    for b in np.array_split(np.arange(len(p)), nb):
        out.append((round(float(p[b].mean()), 6), round(float(y[b].mean()), 6), len(b)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time(); dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    d = np.load(CORPUS, allow_pickle=True)
    act = d['act'].astype(np.int64); seat = d['seat'].astype(np.int64)
    score = d['score'].astype(np.int64); fan = d['fan'].astype(np.int64)
    game = d['game'].astype(np.int64); step = d['step'].astype(np.int64)
    labels, info = derive_labels(game, seat, act, score, fan, step)
    disc = (act >= PLAY) & (act < CHI)
    obs38 = np.ascontiguousarray(d['obs'][disc]).astype(np.int8)       # (Nd,38,4,9)
    act_d = act[disc]
    # candidate-tile plane: one-hot at grid cell of the discarded tile
    idx = act_d - PLAY                                                 # 0..33
    Nd = len(idx); cand = np.zeros((Nd, 1, 4, 9), np.int8)
    cand[np.arange(Nd), 0, idx // 9, idx % 9] = 1
    Xall = torch.from_numpy(np.concatenate([obs38, cand], axis=1))     # (Nd,39,4,9) int8
    del obs38, cand
    y = labels[disc].astype(np.float32); gsub = game[disc]
    nlegal = d['mask'][disc][:, PLAY:CHI].sum(1).astype(np.int32)      # legal discard count
    pos_rate = float(y.mean())
    print(f"[pc s{a.seed}] games={info['games']} ron={info['ron']} zimo={info['zimo']} "
          f"draw={info['draw']} | discard_rows={Nd} pos={int(y.sum())} pos_rate={pos_rate:.4f} "
          f"| rows_with_>1_legal={(nlegal>1).mean():.3f}", flush=True)

    rng = np.random.RandomState(20260713)          # SAME split seed as state model
    ug = np.unique(gsub); rng.shuffle(ug)
    n_val = int(len(ug) * a.val_frac); val_games = set(ug[:n_val].tolist())
    is_val = np.array([g in val_games for g in gsub])
    tr_idx = np.flatnonzero(~is_val); va_idx = np.flatnonzero(is_val)
    assert len(set(gsub[tr_idx].tolist()) & set(gsub[va_idx].tolist())) == 0, "game leak!"
    va_multi = va_idx[nlegal[va_idx] > 1]          # diagnostic subset
    print(f"[pc s{a.seed}] train={len(tr_idx)} (pos {y[tr_idx].mean():.4f}) val={len(va_idx)} "
          f"(pos {y[va_idx].mean():.4f}) val_>1legal={len(va_multi)} "
          f"(pos {y[va_multi].mean():.4f}) val_games={n_val}", flush=True)
    yt_dev = torch.from_numpy(y).to(dev)

    net = DealInNet(a.channels, a.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.1)
    lossf = nn.BCEWithLogitsLoss()
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    from sklearn.metrics import roc_auc_score, average_precision_score

    @torch.no_grad()
    def predict(em, idxs):
        ps = []
        for i in range(0, len(idxs), 8192):
            bi = idxs[i:i + 8192]
            ps.append(torch.sigmoid(em(Xall[bi].to(dev).float())).cpu().numpy())
        return np.concatenate(ps)

    def ema_model():
        em = DealInNet(a.channels, a.blocks).to(dev); sd = em.state_dict()
        for k in sd: sd[k] = ema[k].to(sd[k].dtype)
        em.load_state_dict(sd); em.eval(); return em

    def evaluate():
        em = ema_model()
        pv = predict(em, va_idx); yv = y[va_idx]
        pm = predict(em, va_multi); ym = y[va_multi]
        return (float(roc_auc_score(yv, pv)), float(average_precision_score(yv, pv)),
                float(roc_auc_score(ym, pm)), float(average_precision_score(ym, pm)),
                calib_deciles(yv, pv))

    rs = np.random.RandomState(1000 + a.seed); net.train()
    for step_i in range(1, a.steps + 1):
        bi = tr_idx[rs.randint(0, len(tr_idx), a.bs)]
        logit = net(Xall[bi].to(dev).float()); loss = lossf(logit, yt_dev[bi])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if ema[k].is_floating_point(): ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
                else: ema[k].copy_(v)
        if step_i % 500 == 0:
            print(f"[pc s{a.seed}] step {step_i}/{a.steps} bce={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if step_i % a.val_every == 0 or step_i == a.steps:
            au, apr, aum, aprm, cal = evaluate()
            print(f"[pc s{a.seed}] VAL step {step_i}: AUROC={au:.4f} PR-AUC={apr:.4f} | "
                  f">1legal AUROC={aum:.4f} PR-AUC={aprm:.4f}", flush=True)
            net.train()

    au, apr, aum, aprm, cal = evaluate()
    em = ema_model(); fused = fuse_dealin(em.cpu())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(fused.state_dict(), a.out)
    tinfo = dict(seed=a.seed, arch="DealInFused", in_planes=IN_PLANES, channels=a.channels,
                 blocks=a.blocks, steps=a.steps, bs=a.bs, lr=a.lr, loss="BCE_unweighted",
                 conditioning="per-candidate: 39th one-hot tile plane at ((act-2)//9,(act-2)%9)",
                 label="per-discard ron-causing (act in [2,36))", corpus=CORPUS, games=info,
                 pos_rate=pos_rate, n_train=int(len(tr_idx)), n_val=int(len(va_idx)),
                 n_val_gt1legal=int(len(va_multi)), val_frac=a.val_frac, val_split_seed=20260713,
                 train_sample_seed=1000 + a.seed,
                 val_auroc=round(au, 4), val_prauc=round(apr, 4),
                 val_auroc_gt1legal=round(aum, 4), val_prauc_gt1legal=round(aprm, 4),
                 val_base_rate=round(float(y[va_idx].mean()), 4),
                 val_base_rate_gt1legal=round(float(y[va_multi].mean()), 4),
                 calibration_deciles=[dict(pred=c[0], actual=c[1], n=c[2]) for c in cal],
                 elapsed_s=round(time.time() - t0, 1))
    json.dump(tinfo, open(a.out + ".traininfo.json", "w"), indent=2)
    print(f"[pc s{a.seed}] DONE AUROC={au:.4f} PR-AUC={apr:.4f} | >1legal AUROC={aum:.4f} "
          f"-> {a.out} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[pc s{a.seed}] CALIB(pred,actual,n): {cal}", flush=True)

if __name__ == "__main__":
    main()
