"""
oppdealin_train.py -- OFF-POLICY per-candidate deal-in predictor (v2).

Same architecture/label-conditioning as dealin_pc_train.py (ResBN 128x40 trunk + 39th one-hot
candidate-tile plane -> 1 logit, BCE, EMA, fused deploy ckpt), but trained on OFF-POLICY,
counterfactually-complete data from oppdealin_gen.py: every legal candidate tile T at each
discard decision, labeled by the engine's own Ron gate. Split by GAME.

  CUDA_VISIBLE_DEVICES=3 python3 oppdealin_train.py --seed 0 --steps 50000 \
        --data data/oppdealin/full --out ckpt/dealin_pc_v2/dealin_pc_v2_s0.pt
"""
import os, sys, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

IN_PLANES, GRID = 39, 4 * 9
VAL_SPLIT_SEED = 20260713          # SAME split seed as dealin_pc / state model


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


def load_pairs(data_dir):
    """Load per-decision shards -> flattened per-candidate pairs (dec_idx, tile, label, game)."""
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    assert shards, f"no shards in {data_dir}"
    OBS = []; LEG = []; DI = []; GM = []
    for sp in shards:
        z = np.load(sp)
        if len(z["obs"]) == 0:
            continue
        OBS.append(z["obs"].astype(np.int8)); LEG.append(z["legal"].astype(np.int8))
        DI.append(z["dealin"].astype(np.int8)); GM.append(z["game"].astype(np.int64))
    obs = np.concatenate(OBS); legal = np.concatenate(LEG)
    dealin = np.concatenate(DI); game = np.concatenate(GM)
    dec_idx, tile = np.nonzero(legal)                       # per-candidate expansion
    dec_idx = dec_idx.astype(np.int64); tile = tile.astype(np.int64)
    label = dealin[dec_idx, tile].astype(np.float32)
    pgame = game[dec_idx]
    return obs, dec_idx, tile, label, pgame, game


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
    ap.add_argument("--data", default="data/oppdealin/full")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time(); dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    obs, dec_idx, tile, label, pgame, game = load_pairs(a.data)
    Ndec = len(obs); Np = len(label); pos_rate = float(label.mean())
    print(f"[v2 s{a.seed}] decisions={Ndec} candidate_pairs={Np} pos={int(label.sum())} "
          f"PER_CANDIDATE_BASE_RATE={pos_rate:.4f} games={len(np.unique(game))}", flush=True)

    # game-disjoint split
    rng = np.random.RandomState(VAL_SPLIT_SEED)
    ug = np.unique(game); rng.shuffle(ug)
    n_val = int(len(ug) * a.val_frac); val_games = set(ug[:n_val].tolist())
    is_val = np.array([g in val_games for g in pgame])
    tr_idx = np.flatnonzero(~is_val); va_idx = np.flatnonzero(is_val)
    assert len(set(pgame[tr_idx].tolist()) & set(pgame[va_idx].tolist())) == 0, "GAME LEAK!"
    print(f"[v2 s{a.seed}] train_pairs={len(tr_idx)} (pos {label[tr_idx].mean():.4f}) "
          f"val_pairs={len(va_idx)} (pos {label[va_idx].mean():.4f}) val_games={n_val}", flush=True)

    # keep obs on GPU (int8), build 39-plane inputs per batch
    obs_gpu = torch.from_numpy(obs).to(dev)                 # (Ndec,38,4,9) int8
    dec_t = torch.from_numpy(dec_idx).to(dev)
    tile_t = torch.from_numpy(tile).to(dev)
    lab_t = torch.from_numpy(label).to(dev)
    tr_t = torch.from_numpy(tr_idx).to(dev)

    def build_batch(pair_ids):
        di = dec_t[pair_ids]; ti = tile_t[pair_ids]; n = len(pair_ids)
        o = obs_gpu[di].float()                            # (n,38,4,9)
        cand = torch.zeros(n, 1, 4, 9, device=dev)
        cand[torch.arange(n, device=dev), 0, torch.div(ti, 9, rounding_mode='floor'), ti % 9] = 1.0
        return torch.cat([o, cand], dim=1)                 # (n,39,4,9)

    net = DealInNet(a.channels, a.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.1)
    lossf = nn.BCEWithLogitsLoss()
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    from sklearn.metrics import roc_auc_score, average_precision_score

    def ema_model():
        em = DealInNet(a.channels, a.blocks).to(dev); sd = em.state_dict()
        for k in sd: sd[k] = ema[k].to(sd[k].dtype)
        em.load_state_dict(sd); em.eval(); return em

    @torch.no_grad()
    def predict(em, idxs):
        ps = []
        for i in range(0, len(idxs), 8192):
            bi = torch.from_numpy(idxs[i:i + 8192]).to(dev)
            ps.append(torch.sigmoid(em(build_batch(bi))).cpu().numpy())
        return np.concatenate(ps)

    def evaluate():
        em = ema_model()
        pv = predict(em, va_idx); yv = label[va_idx]
        return (float(roc_auc_score(yv, pv)), float(average_precision_score(yv, pv)),
                calib_deciles(yv, pv))

    rs = np.random.RandomState(1000 + a.seed); net.train()
    ntr = len(tr_t)
    for step_i in range(1, a.steps + 1):
        sel = torch.from_numpy(rs.randint(0, ntr, a.bs)).to(dev)
        pair_ids = tr_t[sel]
        logit = net(build_batch(pair_ids)); loss = lossf(logit, lab_t[pair_ids])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if ema[k].is_floating_point(): ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
                else: ema[k].copy_(v)
        if step_i % 500 == 0:
            print(f"[v2 s{a.seed}] step {step_i}/{a.steps} bce={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)", flush=True)
        if step_i % a.val_every == 0 or step_i == a.steps:
            au, apr, cal = evaluate()
            print(f"[v2 s{a.seed}] VAL step {step_i}: AUROC={au:.4f} PR-AUC={apr:.4f}", flush=True)
            net.train()

    au, apr, cal = evaluate()
    em = ema_model(); fused = fuse_dealin(em.cpu())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(fused.state_dict(), a.out)
    tinfo = dict(seed=a.seed, arch="DealInFused", in_planes=IN_PLANES, channels=a.channels,
                 blocks=a.blocks, steps=a.steps, bs=a.bs, lr=a.lr, loss="BCE_unweighted",
                 conditioning="per-candidate: 39th one-hot tile plane at (tile//9,tile%9)",
                 label="OFF-POLICY: engine _fan Ron gate on every legal candidate vs true opp hands",
                 data=a.data, decisions=int(Ndec), candidate_pairs=int(Np),
                 games=int(len(np.unique(game))), per_candidate_base_rate=round(pos_rate, 6),
                 n_train=int(len(tr_idx)), n_val=int(len(va_idx)), val_frac=a.val_frac,
                 val_split_seed=VAL_SPLIT_SEED, train_sample_seed=1000 + a.seed,
                 val_auroc=round(au, 4), val_prauc=round(apr, 4),
                 val_base_rate=round(float(label[va_idx].mean()), 4),
                 calibration_deciles=[dict(pred=c[0], actual=c[1], n=c[2]) for c in cal],
                 elapsed_s=round(time.time() - t0, 1))
    json.dump(tinfo, open(a.out + ".traininfo.json", "w"), indent=2)
    print(f"[v2 s{a.seed}] DONE AUROC={au:.4f} PR-AUC={apr:.4f} -> {a.out} ({time.time()-t0:.0f}s)", flush=True)
    open(a.out.replace(".pt", ".DONE"), "w").write("ok\n")


if __name__ == "__main__":
    main()
