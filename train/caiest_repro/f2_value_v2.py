"""
f2_value_v2.py — VALUE-HEAD V2 (planned track, Win-2027): final-score value function,
same 128x40 e2e trunk + protocol as f2_value_head.py --mode e2e (v1, results/value_e2e.json
r_all=0.7118 / r_late=0.7771, Final2-only), with the two requested upgrades:

  (1) MORE DATA: add the official 98k-game corpus (cooked_obs.npy rows + per-decision
      final-score labels from data/official_value_labels.npz, alignment-verified).
  (2) SOURCE CONDITIONING: small source embedding (Final2 bot id 0-3; official-human id 4)
      concatenated before the value head.

Variants:
  a  v1-repro   : Final2 only, no conditioning (baseline; v1 protocol, batch 512)
  b  all-data   : Final2 + official, 50/50 per batch, no conditioning
  c  all+cond   : as b + source embedding

Split BY GAME, rng 777, 10% held-out — IDENTICAL split code to v1 for Final2; same rule
applied independently to official games. Metrics identical to v1 (metrics() copied verbatim):
Pearson r / Spearman rho overall+by stage + GRP-style within-game 4-player rank rho, computed
separately on the Final2 held-out set and the official held-out set (cross-domain for a).
Score scale SC=30 for both domains (v1 convention). 30k steps, AdamW 3e-4/wd 1e-4, cosine,
AMP, exactly as v1.

  python3 f2_value_v2.py --variant a --seed 0 --out results/value_v2_a_s0.json
"""
import os, sys, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as F
from torch import nn
from models_explore import ResBNCNN

HERE = os.path.dirname(os.path.abspath(__file__))
SC = 30.0
F2_CORPUS = "/root/final2_harvest/final2_cai_corpus.npz"
OF_OBS = "/root/IJCAI-mahjong-full/IJCAI-mahjong/train/caiest_repro/data/cooked_obs.npy"
OF_LABELS = os.path.join(HERE, "data", "official_value_labels.npz")
OFFICIAL_SRC_ID = 4   # Final2 bots are 0-3


# ---- metrics: verbatim from f2_value_head.py (v1) ----
def rankdata(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def stage_of(step, gamelen):
    fr = step.astype(np.float64) / np.maximum(gamelen - 1, 1)
    st = np.full(len(fr), 1, np.int8)
    st[fr < 1.0 / 3] = 0
    st[fr >= 2.0 / 3] = 2
    return st


def metrics(pred, y, game, seat, stage):
    out = {}
    out["r_all"] = round(pearson(pred, y), 4)
    out["rho_all"] = round(spearman(pred, y), 4)
    names = ["early", "mid", "late"]
    for k in range(3):
        s = stage == k
        out[f"r_{names[k]}"] = round(pearson(pred[s], y[s]), 4) if s.sum() > 10 else None
        out[f"rho_{names[k]}"] = round(spearman(pred[s], y[s]), 4) if s.sum() > 10 else None
    for k in range(3):
        s = np.flatnonzero(stage == k)
        key = game[s].astype(np.int64) * 4 + seat[s].astype(np.int64)
        sums = {}
        cnts = {}
        for i, kk in zip(s, key):
            sums[kk] = sums.get(kk, 0.0) + pred[i]
            cnts[kk] = cnts.get(kk, 0) + 1
        rhos = []
        by_game = {}
        for kk in sums:
            by_game.setdefault(kk // 4, []).append(kk % 4)
        truth = {}
        for i, kk in zip(s, key):
            truth[kk] = y[i]
        for g, seats in by_game.items():
            if len(seats) < 4:
                continue
            pm = [sums[g * 4 + q] / cnts[g * 4 + q] for q in range(4)]
            tm = [truth[g * 4 + q] for q in range(4)]
            rhos.append(spearman(pm, tm))
        out[f"grp_rank_rho_{names[k]}"] = round(float(np.mean(rhos)), 4) if rhos else None
        out[f"grp_games_{names[k]}"] = len(rhos)
    return out


def split_val_games(game, frac=0.1, seed=777):
    """v1 split, verbatim: rng 777, 10% of unique games held out."""
    ug = np.unique(game)
    rng = np.random.RandomState(seed)
    vg = set(ug[rng.permutation(len(ug))[:len(ug) // 10]].tolist())
    is_val = np.isin(game, list(vg))
    return np.flatnonzero(is_val), np.flatnonzero(~is_val)


class VNet(nn.Module):
    """v1 e2e trunk (ResBNCNN 128x40 stem+body) + value foot; optional source embedding
    concatenated before the final linear."""
    def __init__(self, cond=False, n_src=5, emb_dim=16):
        super().__init__()
        base = ResBNCNN(channels=128, blocks=40)
        self.stem, self.body = base.stem, base.body
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(128 * 36, 512)
        self.cond = cond
        if cond:
            self.emb = nn.Embedding(n_src, emb_dim)
            self.head = nn.Linear(512 + emb_dim, 1)
        else:
            self.head = nn.Linear(512, 1)

    def forward(self, x, src=None):
        h = F.relu(self.fc1(self.flat(self.body(self.stem(x)))))
        if self.cond:
            h = torch.cat([h, self.emb(src)], dim=1)
        return self.head(h).squeeze(1)


def predict(net, obs_arr, idx, src_arr, dev, cond):
    preds = np.empty(len(idx), np.float32)
    net.eval()
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]
            ob = torch.from_numpy(np.ascontiguousarray(obs_arr[b])).float().to(dev)
            sb = torch.from_numpy(src_arr[b].astype(np.int64)).to(dev) if cond else None
            with torch.cuda.amp.autocast():
                preds[i:i + 4096] = net(ob, sb).float().cpu().numpy()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["a", "b", "c"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    cond = a.variant == "c"
    use_official_train = a.variant in ("b", "c")

    # ---- Final2 corpus (v1 data) ----
    d = np.load(F2_CORPUS)
    f2_obs = d["obs"]
    f2_y = d["score"].astype(np.float32) / SC
    f2_game, f2_seat, f2_bot = d["game"], d["seat"], d["bot"].astype(np.int64)
    f2_stage = stage_of(d["step"], d["gamelen"])
    f2_vidx, f2_tidx = split_val_games(f2_game)
    print(f"final2 N={len(f2_y):,} train={len(f2_tidx):,} val={len(f2_vidx):,}", flush=True)

    # ---- official corpus (labels verified against cooked_act.npy) ----
    lab = np.load(OF_LABELS)
    assert int(lab["verified"]) == 1, "official labels not alignment-verified"
    of_y = lab["score"].astype(np.float32) / SC
    of_game, of_seat = lab["game"], lab["seat"]
    of_stage = stage_of(lab["step"], lab["gamelen"])
    of_vidx, of_tidx = split_val_games(of_game)
    mm = np.load(OF_OBS, mmap_mode="r")
    assert len(mm) == len(of_y)
    if use_official_train:
        print("loading full official obs into RAM ...", flush=True)
        of_obs = np.asarray(mm)               # 8 GB int8
        of_obs_val, of_val_map = of_obs, of_vidx
    else:
        of_obs_val = np.ascontiguousarray(mm[of_vidx])   # val rows only (cross-domain eval)
        of_val_map = np.arange(len(of_vidx))
    print(f"official N={len(of_y):,} train={len(of_tidx):,} val={len(of_vidx):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    f2_src = f2_bot                                       # 0..3
    of_src = np.full(len(of_y), OFFICIAL_SRC_ID, np.int64)

    # ---- model / optimizer: exactly v1's recipe ----
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    net = VNet(cond=cond).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.cuda.amp.GradScaler()
    rb = np.random.RandomState(2000 + a.seed)

    net.train()
    for s in range(a.steps):
        if use_official_train:
            b1 = f2_tidx[np.sort(rb.randint(0, len(f2_tidx), 256))]
            b2 = of_tidx[np.sort(rb.randint(0, len(of_tidx), 256))]
            ob = torch.from_numpy(np.concatenate(
                [np.ascontiguousarray(f2_obs[b1]), np.ascontiguousarray(of_obs[b2])])).float().to(dev)
            yb = torch.from_numpy(np.concatenate([f2_y[b1], of_y[b2]])).to(dev)
            sb = torch.from_numpy(np.concatenate([f2_src[b1], of_src[b2]])).to(dev) if cond else None
        else:
            bi = f2_tidx[np.sort(rb.randint(0, len(f2_tidx), 512))]
            ob = torch.from_numpy(np.ascontiguousarray(f2_obs[bi])).float().to(dev)
            yb = torch.from_numpy(f2_y[bi]).to(dev)
            sb = torch.from_numpy(f2_src[bi]).to(dev) if cond else None
        with torch.cuda.amp.autocast():
            loss = F.mse_loss(net(ob, sb).float(), yb)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt)
        scaler.update(); sch.step()
        if s % 2000 == 0:
            print(f"  step {s}/{a.steps} mse {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    # save deployable checkpoint (search leaf evaluator for 2027)
    import torch as _t
    _ck = a.out[:-5] + '.pt' if a.out.endswith('.json') else a.out + '.pt'
    _t.save(net.state_dict(), _ck)
    print('SAVED_CKPT ' + _ck, flush=True)

    # ---- eval: Final2 held-out + official held-out, v1 metrics ----
    pf = predict(net, f2_obs, f2_vidx, f2_src, dev, cond)
    m_f2 = metrics(pf, f2_y[f2_vidx], f2_game[f2_vidx], f2_seat[f2_vidx], f2_stage[f2_vidx])
    m_f2["val_mse"] = round(float(np.mean((pf - f2_y[f2_vidx]) ** 2)), 4)
    print("FINAL2 " + json.dumps(m_f2), flush=True)

    po = predict(net, of_obs_val, of_val_map, of_src, dev, cond)
    m_of = metrics(po, of_y[of_vidx], of_game[of_vidx], of_seat[of_vidx], of_stage[of_vidx])
    m_of["val_mse"] = round(float(np.mean((po - of_y[of_vidx]) ** 2)), 4)
    print("OFFICIAL " + json.dumps(m_of), flush=True)

    out = dict(mode="value_v2_e2e_resbn128x40", variant=a.variant, seed=a.seed,
               steps=a.steps, score_scale=SC, conditioning=cond,
               train_data=("final2_only" if not use_official_train else "final2+official_50_50"),
               n_val_final2=int(len(f2_vidx)), n_val_official=int(len(of_vidx)),
               src_ids="final2 bot 0-3, official-human 4" if cond else None,
               metrics_final2=m_f2, metrics_official=m_of,
               seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("DONE " + a.out, flush=True)


if __name__ == "__main__":
    main()
