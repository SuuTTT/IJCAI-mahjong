"""
f2_value_head.py — Exp2: SCORE-VALUE HEAD on the Final2 finalist corpus.

Predict the ACTOR's final game score from the state (corpus has final duplicate score per
decision). Two modes:
  --mode frozen : frozen kdens student trunk (kd_128x40_s0.bn.pkl, penultimate 512-d
                  features precomputed once) + MLP head (512-256-1), MSE, 3 seeds in-job.
  --mode e2e    : full ResBNCNN 128x40 trained from scratch with a 1-d value foot
                  (single seed, strength reference).

Split is BY GAME (rng 777, 10% held-out games). Metrics on held-out games:
  * Pearson r (pred vs final score), overall and by stage (early/mid/late thirds of the
    game's decision sequence)
  * Spearman rho, overall and by stage
  * GRP-style within-game 4-player rank correlation by stage: per game+stage, average
    prediction per seat vs the true 4 final scores, Spearman, averaged over games.
    (in-house GRP prior: r = 0.829 on real data)

frozen -> results/value_frozen.json ; e2e -> results/value_e2e.json
"""
import os, sys, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as F
from torch import nn
from models_explore import ResBNCNN

HERE = os.path.dirname(os.path.abspath(__file__))
SC = 30.0  # score normalizer


def rankdata(x):
    """average-tie ranks, 1-based (numpy only)."""
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
    # GRP-style within-game 4-player rank corr per stage
    for k in range(3):
        s = np.flatnonzero(stage == k)
        key = game[s].astype(np.int64) * 4 + seat[s].astype(np.int64)
        sums = {}
        cnts = {}
        for i, kk in zip(s, key):
            sums[kk] = sums.get(kk, 0.0) + pred[i]
            cnts[kk] = cnts.get(kk, 0) + 1
        # true score per (game,seat)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "e2e"], required=True)
    ap.add_argument("--corpus", default="/root/final2_harvest/final2_cai_corpus.npz")
    ap.add_argument("--trunk", default="ckpt/kd/kd_128x40_s0.bn.pkl")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=12000)      # frozen head steps
    ap.add_argument("--e2e_steps", type=int, default=30000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    d = np.load(a.corpus)
    obs, y = d["obs"], (d["score"].astype(np.float32) / SC)
    game, seat = d["game"], d["seat"]
    stage = stage_of(d["step"], d["gamelen"])
    N = len(y)
    ug = np.unique(game)
    rng = np.random.RandomState(777)
    vg = set(ug[rng.permutation(len(ug))[:len(ug) // 10]].tolist())
    is_val = np.isin(game, list(vg))
    vidx = np.flatnonzero(is_val); tidx = np.flatnonzero(~is_val)
    print(f"N={N:,} train={len(tidx):,} val={len(vidx):,} ({len(vg)} held-out games)", flush=True)

    if a.mode == "frozen":
        trunk = ResBNCNN(channels=128, blocks=40)
        trunk.load_state_dict(torch.load(a.trunk, map_location="cpu"))
        trunk.eval().to(dev)
        for p in trunk.parameters():
            p.requires_grad_(False)
        feat_net = nn.Sequential(trunk.stem, trunk.body, trunk.foot[0], trunk.foot[1],
                                 trunk.foot[2]).eval()
        # precompute 512-d features (fp16, CPU RAM)
        feats = np.empty((N, 512), np.float16)
        with torch.no_grad():
            for i in range(0, N, 4096):
                ob = torch.from_numpy(np.ascontiguousarray(obs[i:i + 4096])).float().to(dev)
                with torch.cuda.amp.autocast():
                    f_ = feat_net(ob)
                feats[i:i + 4096] = f_.half().cpu().numpy()
                if i % (4096 * 100) == 0:
                    print(f"  feat {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
        print(f"features done ({time.time()-t0:.0f}s)", flush=True)

        yv = y[vidx]
        per_seed = []
        for sd in [int(x) for x in a.seeds.split(",")]:
            torch.manual_seed(sd)
            head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)
            opt = torch.optim.Adam(head.parameters(), lr=1e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
            rb = np.random.RandomState(1000 + sd)
            head.train()
            for s in range(a.steps):
                b = tidx[rb.randint(0, len(tidx), 4096)]
                fb = torch.from_numpy(feats[b]).float().to(dev)
                yb = torch.from_numpy(y[b]).to(dev)
                loss = F.mse_loss(head(fb).squeeze(1), yb)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            head.eval()
            preds = np.empty(len(vidx), np.float32)
            with torch.no_grad():
                for i in range(0, len(vidx), 16384):
                    b = vidx[i:i + 16384]
                    fb = torch.from_numpy(feats[b]).float().to(dev)
                    preds[i:i + 16384] = head(fb).squeeze(1).cpu().numpy()
            mm = metrics(preds, yv, game[vidx], seat[vidx], stage[vidx])
            mm["seed"] = sd
            mm["val_mse"] = round(float(np.mean((preds - yv) ** 2)), 4)
            per_seed.append(mm)
            print(json.dumps(mm), flush=True)
        agg = dict(mode="frozen_trunk+mlp_head", trunk=os.path.basename(a.trunk),
                   head="512-256-1", steps=a.steps, n_val=int(len(vidx)),
                   score_scale=SC, per_seed=per_seed,
                   r_all_mean=round(float(np.mean([m["r_all"] for m in per_seed])), 4),
                   seconds=round(time.time() - t0, 1))
        with open(a.out, "w") as f:
            json.dump(agg, f, indent=2)
        print("DONE frozen", flush=True)

    else:  # e2e
        sd = int(a.seeds.split(",")[0])
        torch.manual_seed(sd); np.random.seed(sd)
        net = ResBNCNN(channels=128, blocks=40).to(dev)
        net.foot = nn.Sequential(nn.Flatten(), nn.Linear(128 * 36, 512), nn.ReLU(),
                                 nn.Linear(512, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.e2e_steps)
        scaler = torch.cuda.amp.GradScaler()
        rb = np.random.RandomState(2000 + sd)
        ones = None
        for s in range(a.e2e_steps):
            b = np.sort(rb.randint(0, len(tidx), 512))
            bi = tidx[b]
            ob = torch.from_numpy(np.ascontiguousarray(obs[bi])).float().to(dev)
            yb = torch.from_numpy(y[bi]).to(dev)
            with torch.cuda.amp.autocast():
                x = net.foot(net.body(net.stem(ob))).squeeze(1)
                loss = F.mse_loss(x.float(), yb)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt)
            scaler.update(); sch.step()
            if s % 2000 == 0:
                print(f"  e2e step {s}/{a.e2e_steps} mse {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        net.eval()
        preds = np.empty(len(vidx), np.float32)
        with torch.no_grad():
            for i in range(0, len(vidx), 4096):
                b = vidx[i:i + 4096]
                ob = torch.from_numpy(np.ascontiguousarray(obs[b])).float().to(dev)
                with torch.cuda.amp.autocast():
                    preds[i:i + 4096] = net.foot(net.body(net.stem(ob))).squeeze(1).float().cpu().numpy()
        yv = y[vidx]
        mm = metrics(preds, yv, game[vidx], seat[vidx], stage[vidx])
        mm["val_mse"] = round(float(np.mean((preds - yv) ** 2)), 4)
        agg = dict(mode="e2e_resbn128x40_value", seed=sd, steps=a.e2e_steps,
                   n_val=int(len(vidx)), score_scale=SC, metrics=mm,
                   seconds=round(time.time() - t0, 1))
        with open(a.out, "w") as f:
            json.dump(agg, f, indent=2)
        torch.save(net.state_dict(), a.out.replace(".json", ".pt"))
        print("SAVED " + a.out.replace(".json", ".pt"), flush=True)
        print("DONE e2e " + json.dumps(mm), flush=True)


if __name__ == "__main__":
    main()
