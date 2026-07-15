"""
oppbelief_train.py -- OPPONENT HAND-BELIEF model.

Trunk = ResBN 128x40 (models_explore), head -> 3*34=102 logits = P(rel-opponent holds >=1
of tile j). Binary multi-label BCE, EMA, fused deploy ckpt. Deploy interface: given seat s's
38-plane public obs -> sigmoid -> (3,34) held-probability array (rel 1=next,2=across,3=prev).

Eval (held-out games, game-disjoint split):
  * per-tile AUROC + Brier over entries that COULD be held (unseen count r_j>0).
  * THE key metric: mean log-likelihood of the ACTUAL holdings under the model vs under the
    UNIFORM baseline (deal the r_j unseen copies at random among the U unseen tiles; opp holds
    h of them -> hypergeometric P(hold>=1)=1-C(U-r,h)/C(U,h)). Model must beat uniform.
  * tile-count constraint respected IN EVAL ONLY: model prob zeroed where r_j==0 (can't hold).

  CUDA_VISIBLE_DEVICES=0 python3 oppbelief_train.py --tag full --seed 0 --steps 50000 \
     --out ckpt/oppbelief/oppbelief_s0.pt
"""
import os, sys, json, time, argparse, glob
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from scipy.special import gammaln

IN_PLANES, GRID, NOUT = 38, 4 * 9, 3 * 34
DATADIR = "/root/caiest_repro/data/oppbelief"


class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return torch.relu(x + y)

class BeliefNet(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU(),
                                  nn.Linear(512, NOUT))
    def forward(self, x):
        return self.foot(self.body(self.stem(x)))          # (B,102) logits

class _FusedBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True); self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=True)
    def forward(self, x):
        y = torch.relu(self.c1(x)); y = self.c2(y); return torch.relu(x + y)

class BeliefFused(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__()
        self.stem = nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*(_FusedBlock(channels) for _ in range(blocks)))
        self.foot = nn.Sequential(nn.Flatten(), nn.Linear(channels * GRID, 512), nn.ReLU(),
                                  nn.Linear(512, NOUT))
    def forward(self, x):
        return self.foot(self.body(torch.relu(self.stem(x))))

def fuse_belief(net):
    net.eval(); ch = net.stem[0].out_channels; blocks = len(net.body)
    f = BeliefFused(ch, blocks).eval()
    f.stem.load_state_dict(fuse_conv_bn_eval(net.stem[0], net.stem[1]).state_dict())
    for i, blk in enumerate(net.body):
        f.body[i].c1.load_state_dict(fuse_conv_bn_eval(blk.c1, blk.b1).state_dict())
        f.body[i].c2.load_state_dict(fuse_conv_bn_eval(blk.c2, blk.b2).state_dict())
    f.foot.load_state_dict(net.foot.state_dict())
    return f


def load_shards(tag):
    fs = sorted(glob.glob(os.path.join(DATADIR, tag, "shard_*.npz")))
    assert fs, f"no shards for tag {tag}"
    O, T, U, H, G = [], [], [], [], []
    for f in fs:
        d = np.load(f)
        O.append(d["obs"]); T.append(d["tgt"]); U.append(d["uns"]); H.append(d["hsz"]); G.append(d["game"])
    return (np.concatenate(O), np.concatenate(T), np.concatenate(U),
            np.concatenate(H), np.concatenate(G))


def logC(n, k):
    """log C(n,k) elementwise; 0 (i.e. C=0) where n<k or k<0."""
    n = n.astype(np.float64); k = k.astype(np.float64)
    out = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    out[(n < k) | (k < 0)] = -np.inf                        # C=0 -> log=-inf
    return out

def uniform_phold(uns, hsz):
    """uns:(N,34) r_j ; hsz:(N,3) h. Return p_unif:(N,3,34)=1-C(U-r,h)/C(U,h)."""
    N = len(uns); U = uns.sum(1).astype(np.float64)                    # (N,)
    r = uns.astype(np.float64)                                         # (N,34)
    # broadcast to (N,3,34)
    Ub = U[:, None, None]; rb = r[:, None, :]; hb = hsz.astype(np.float64)[:, :, None]
    num = logC(Ub - rb, np.broadcast_to(hb, (N, 3, 34)))
    den = logC(np.broadcast_to(Ub, (N, 3, 34)), np.broadcast_to(hb, (N, 3, 34)))
    ratio = np.exp(np.clip(num - den, -700, 0))                        # C(U-r,h)/C(U,h) in [0,1]
    ratio[np.isneginf(num)] = 0.0                                      # opp must hold it
    return 1.0 - ratio


def bern_ll(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
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
    from sklearn.metrics import roc_auc_score

    obs, tgt, uns, hsz, game = load_shards(a.tag)
    N = len(obs)
    y = (tgt >= 1).astype(np.float32).reshape(N, NOUT)                 # (N,102) hold>=1
    Xall = torch.from_numpy(np.ascontiguousarray(obs)).to(torch.int8)
    print(f"[s{a.seed}] loaded {N} samples, {len(np.unique(game))} games, "
          f"base_hold_rate(all)={y.mean():.4f}", flush=True)

    rng = np.random.RandomState(20260714)
    ug = np.unique(game); rng.shuffle(ug)
    nval = int(len(ug) * a.val_frac); vg = set(ug[:nval].tolist())
    is_val = np.array([g in vg for g in game])
    tr = np.flatnonzero(~is_val); va = np.flatnonzero(is_val)
    assert len(set(game[tr].tolist()) & set(game[va].tolist())) == 0, "game leak!"
    print(f"[s{a.seed}] train={len(tr)} val={len(va)} val_games={nval}", flush=True)
    yt = torch.from_numpy(y).to(dev)

    # precompute uniform baseline on val
    uns_va = uns[va]; hsz_va = hsz[va]
    p_unif = uniform_phold(uns_va, hsz_va).reshape(len(va), NOUT)      # (Nv,102)
    unseen_mask = (np.repeat(uns_va, 3, axis=0).reshape(len(va), 3, 34) > 0).reshape(len(va), NOUT)
    yv = y[va]
    ll_unif = bern_ll(yv[unseen_mask], p_unif[unseen_mask])
    brier_unif = float(np.mean((p_unif[unseen_mask] - yv[unseen_mask]) ** 2))
    auroc_unif = roc_auc_score(yv[unseen_mask], p_unif[unseen_mask])
    print(f"[s{a.seed}] UNIFORM baseline (uns>0): AUROC={auroc_unif:.4f} "
          f"Brier={brier_unif:.4f} LL={ll_unif:.4f} base={yv[unseen_mask].mean():.4f}", flush=True)

    net = BeliefNet(a.channels, a.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.1)
    lossf = nn.BCEWithLogitsLoss()
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}

    def ema_model():
        em = BeliefNet(a.channels, a.blocks).to(dev); sd = em.state_dict()
        for k in sd: sd[k] = ema[k].to(sd[k].dtype)
        em.load_state_dict(sd); em.eval(); return em

    @torch.no_grad()
    def predict(em, idxs):
        ps = []
        for i in range(0, len(idxs), 8192):
            xb = Xall[idxs[i:i + 8192]].to(dev).float()
            ps.append(torch.sigmoid(em(xb)).cpu().numpy())
        return np.concatenate(ps)

    def evaluate():
        em = ema_model(); p = predict(em, va)                          # (Nv,102)
        pc = p * unseen_mask                                           # constrain: r_j==0 -> 0
        au = roc_auc_score(yv[unseen_mask], p[unseen_mask])
        br = float(np.mean((pc[unseen_mask] - yv[unseen_mask]) ** 2))
        ll = bern_ll(yv[unseen_mask], pc[unseen_mask])
        return au, br, ll

    rs = np.random.RandomState(1000 + a.seed); net.train()
    for step in range(1, a.steps + 1):
        bi = tr[rs.randint(0, len(tr), a.bs)]
        logit = net(Xall[bi].to(dev).float()); loss = lossf(logit, yt[bi])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if ema[k].is_floating_point(): ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
                else: ema[k].copy_(v)
        if step % 500 == 0:
            print(f"[s{a.seed}] step {step}/{a.steps} bce={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if step % a.val_every == 0 or step == a.steps:
            au, br, ll = evaluate()
            print(f"[s{a.seed}] VAL step {step}: AUROC={au:.4f} Brier={br:.4f} LL={ll:.4f} "
                  f"| UNIF AUROC={auroc_unif:.4f} Brier={brier_unif:.4f} LL={ll_unif:.4f} "
                  f"| beats_unif={ll > ll_unif}", flush=True)
            net.train()

    au, br, ll = evaluate()
    em = ema_model(); fused = fuse_belief(em.cpu())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(fused.state_dict(), a.out)
    info = dict(seed=a.seed, arch="BeliefFused", in_planes=IN_PLANES, out=NOUT,
                target="P(rel-opp holds>=1 of tile j), rel 1=next,2=across,3=prev; BCE",
                channels=a.channels, blocks=a.blocks, steps=a.steps, bs=a.bs, lr=a.lr,
                tag=a.tag, n_samples=int(N), n_train=int(len(tr)), n_val=int(len(va)),
                val_games=nval, val_split_seed=20260714, train_sample_seed=1000 + a.seed,
                base_hold_rate_unsgt0=round(float(yv[unseen_mask].mean()), 4),
                val_auroc=round(au, 4), val_brier=round(br, 4), val_loglik=round(ll, 4),
                uniform_auroc=round(auroc_unif, 4), uniform_brier=round(brier_unif, 4),
                uniform_loglik=round(ll_unif, 4),
                beats_uniform_loglik=bool(ll > ll_unif),
                ll_improvement_over_uniform=round(ll - ll_unif, 4),
                elapsed_s=round(time.time() - t0, 1))
    json.dump(info, open(a.out + ".traininfo.json", "w"), indent=2)
    print(f"[s{a.seed}] DONE AUROC={au:.4f} LL={ll:.4f} (unif {ll_unif:.4f}) "
          f"beats_unif={ll>ll_unif} -> {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
