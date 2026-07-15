"""
dealin_train.py -- DEAL-IN PREDICTOR (defense / threat model) for CSM bot.

Given a discard-decision state (38,4,9 cai obs), predict P(this chosen discard gets Ron'd
-- i.e. ends the hand as the winning tile off this seat's discard). Binary, per-DECISION.

Trunk = ResBNCNN 128x40 (models_explore) with a 1-logit head. Trained with plain BCE
(unweighted -> best calibration; AUROC/PR-AUC are ranking metrics, weighting-invariant).
EMA weights, fused (Conv+BN folded) deploy checkpoint saved.

Labels derived from Final2 corpus: per game (one hand), a Ron is a game with a winner
(fan>0) and a UNIQUE min-scoring loser (score=-(8+fan) < the two others at -8). That loser
is the discarder; their LAST discard row (max step, act in [Play,Chi)=[2,36)) is label 1.
Zimo (all 3 losers tie at -(8+fan)) and draws (no winner) contribute only negatives.

Usage:
  CUDA_VISIBLE_DEVICES=4 python3 dealin_train.py --seed 0 --steps 50000 --out ckpt/dealin/dealin_s0.pt
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

IN_PLANES, GRID = 38, 4 * 9
PLAY, CHI = 2, 36
CORPUS = "/root/final2_harvest/final2_cai_corpus.npz"

# ------------------------- model (ResBN trunk, 1-logit head) -------------------------
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
    def forward(self, x):  # x: (B,38,4,9) float
        return self.foot(self.body(self.stem(x))).squeeze(-1)

class _FusedBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True); self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
    def forward(self, x):
        y = torch.relu(self.c1(x)); y = self.c2(y); return torch.relu(x + y)

class DealInFused(nn.Module):
    """BN-free deploy version (Conv+BN folded). eval-mode identical to DealInNet."""
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

# ------------------------- label derivation -------------------------
def derive_labels(game, seat, act, score, fan, step):
    """Return per-row int8 label: 1 iff row is the ron-causing (winning-tile) discard."""
    N = len(act); labels = np.zeros(N, np.int8)
    order = np.argsort(game, kind='stable'); gs = game[order]
    ug, first = np.unique(gs, return_index=True); bnd = np.append(first, N)
    ron = zimo = draw = 0
    for gi in range(len(ug)):
        rows = order[bnd[gi]:bnd[gi + 1]]
        ss = seat[rows]; sc = score[rows]; fa = fan[rows]; st = step[rows]; aa = act[rows]
        seats = np.unique(ss)
        sscore = {s: sc[ss == s][0] for s in seats}
        sfan = {s: fa[ss == s].max() for s in seats}
        winners = [s for s in seats if sfan[s] > 0]
        if not winners:
            draw += 1; continue
        w = winners[0]
        losers = {s: sscore[s] for s in seats if s != w}
        if not losers: continue
        mn = min(losers.values()); mins = [s for s, v in losers.items() if v == mn]
        if len(mins) == 1 and len(set(losers.values())) > 1:   # unique strict min -> Ron
            ron += 1; disc = mins[0]
            dm = (ss == disc) & (aa >= PLAY) & (aa < CHI)
            if dm.any():
                dr = rows[dm]; labels[dr[np.argmax(st[dm])]] = 1
        else:
            zimo += 1
    return labels, dict(ron=ron, zimo=zimo, draw=draw, games=len(ug))

# ------------------------- metrics -------------------------
def calib_deciles(y, p, nb=10):
    idx = np.argsort(p); y = y[idx]; p = p[idx]
    out = []
    for b in np.array_split(np.arange(len(p)), nb):
        out.append((round(float(p[b].mean()), 6), round(float(y[b].mean()), 6), len(b)))
    return out

# ------------------------- main -------------------------
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
    obs = d['obs']                                  # (N,38,4,9) int8, memmap-ish
    Xall = torch.from_numpy(np.ascontiguousarray(obs[disc])).to(torch.int8)  # CPU
    y = labels[disc].astype(np.float32); gsub = game[disc]
    pos_rate = float(y.mean())
    print(f"[seed{a.seed}] games={info['games']} ron={info['ron']} zimo={info['zimo']} "
          f"draw={info['draw']} | discard_rows={disc.sum()} pos={int(y.sum())} "
          f"pos_rate={pos_rate:.4f}", flush=True)

    # split by GAME (fixed RNG, SAME across seeds so val is comparable + game-disjoint)
    rng = np.random.RandomState(20260713)
    ug = np.unique(gsub); rng.shuffle(ug)
    n_val = int(len(ug) * a.val_frac); val_games = set(ug[:n_val].tolist())
    is_val = np.array([g in val_games for g in gsub])
    tr_idx = np.flatnonzero(~is_val); va_idx = np.flatnonzero(is_val)
    assert len(set(gsub[tr_idx].tolist()) & set(gsub[va_idx].tolist())) == 0, "game leak!"
    print(f"[seed{a.seed}] train={len(tr_idx)} (pos {y[tr_idx].mean():.4f}) "
          f"val={len(va_idx)} (pos {y[va_idx].mean():.4f}) val_games={n_val}", flush=True)
    yt_dev = torch.from_numpy(y).to(dev)

    net = DealInNet(a.channels, a.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.1)
    lossf = nn.BCEWithLogitsLoss()
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}

    from sklearn.metrics import roc_auc_score, average_precision_score

    @torch.no_grad()
    def evaluate():
        em = DealInNet(a.channels, a.blocks).to(dev)
        sd = em.state_dict()
        for k in sd:
            sd[k] = ema[k].to(sd[k].dtype)
        em.load_state_dict(sd); em.eval()
        ps = []
        for i in range(0, len(va_idx), 8192):
            bi = va_idx[i:i + 8192]
            xb = Xall[bi].to(dev).float()
            ps.append(torch.sigmoid(em(xb)).cpu().numpy())
        p = np.concatenate(ps); yv = y[va_idx]
        return (float(roc_auc_score(yv, p)), float(average_precision_score(yv, p)),
                calib_deciles(yv, p), p)

    rs = np.random.RandomState(1000 + a.seed)
    net.train(); best = {}
    for step_i in range(1, a.steps + 1):
        bi = tr_idx[rs.randint(0, len(tr_idx), a.bs)]
        xb = Xall[bi].to(dev).float(); yb = yt_dev[bi]
        logit = net(xb); loss = lossf(logit, yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if ema[k].is_floating_point(): ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
                else: ema[k].copy_(v)
        if step_i % 500 == 0:
            print(f"[seed{a.seed}] step {step_i}/{a.steps} bce={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if step_i % a.val_every == 0 or step_i == a.steps:
            auroc, ap_, cal, _ = evaluate()
            print(f"[seed{a.seed}] VAL step {step_i}: AUROC={auroc:.4f} PR-AUC={ap_:.4f} "
                  f"(base {y[va_idx].mean():.4f})", flush=True)
            best = dict(step=step_i, auroc=auroc, prauc=ap_, calib=cal)
            net.train()

    # final eval + save fused EMA deploy checkpoint
    auroc, ap_, cal, _ = evaluate()
    em = DealInNet(a.channels, a.blocks).to(dev); sd = em.state_dict()
    for k in sd: sd[k] = ema[k].to(sd[k].dtype)
    em.load_state_dict(sd)
    fused = fuse_dealin(em.cpu())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(fused.state_dict(), a.out)
    tinfo = dict(seed=a.seed, arch="DealInFused", channels=a.channels, blocks=a.blocks,
                 steps=a.steps, bs=a.bs, lr=a.lr, loss="BCE_unweighted",
                 label="per-discard ron-causing (act in [2,36))",
                 corpus=CORPUS, games=info, pos_rate=pos_rate,
                 n_train=int(len(tr_idx)), n_val=int(len(va_idx)),
                 val_frac=a.val_frac, val_split_seed=20260713, train_sample_seed=1000 + a.seed,
                 val_auroc=round(auroc, 4), val_prauc=round(ap_, 4),
                 val_base_rate=round(float(y[va_idx].mean()), 4),
                 calibration_deciles=[dict(pred=c[0], actual=c[1], n=c[2]) for c in cal],
                 elapsed_s=round(time.time() - t0, 1))
    json.dump(tinfo, open(a.out + ".traininfo.json", "w"), indent=2)
    print(f"[seed{a.seed}] DONE AUROC={auroc:.4f} PR-AUC={ap_:.4f} -> {a.out} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[seed{a.seed}] CALIB(pred,actual,n):", cal, flush=True)

if __name__ == "__main__":
    main()
