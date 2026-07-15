"""
placeval_train.py — PLACEMENT-predicting value head (PIMC leaf evaluator).

Reuses f2_value_v2's VNet (ResBNCNN 128x40 trunk + value foot) and its verbatim
metrics()/split_val_games(), but regresses the seat's game-final PLACEMENT instead of
final score.

TARGET (higher = better-for-us, range 1..4, mean 2.5):
    for each game, take the 4 seats' final scores (constant within a (game,seat)); then
    placement = 5 - (n_greater + (n_equal+1)/2)   [same average-rank tie rule as the gate]
    Every decision row inherits its seat's game-final placement.
    (1st place -> 4.0, last -> 1.0. Per-game placements sum to 10.)

Model: VNet(cond=False) — regress raw placement with MSE. Net output IS the predicted
placement (higher=better), so deploy is net(obs38[, src]) -> predicted placement directly
(src accepted but ignored when cond=False). We keep an EMA of the weights and both eval
and save the EMA copy as the deploy ckpt.

Data: /root/final2_harvest/final2_cai_corpus.npz (Final2 only — the corpus named in the
task; contains all 4 seats for every game, required for placement). Split BY GAME, rng 777,
10% held out (identical split_val_games as v1). train/val game-disjointness asserted.

  python3 placeval_train.py --seed 0 --steps 30000 --gpu 0 --out results/placeval_s0.json
"""
import os, sys, argparse, time, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as F
# reuse VNet + verbatim metrics/split from the score head
from f2_value_v2 import VNet, metrics, stage_of, split_val_games, pearson, spearman

HERE = os.path.dirname(os.path.abspath(__file__))
F2_CORPUS = "/root/final2_harvest/final2_cai_corpus.npz"


def compute_placement(game, seat, score):
    """Per-row game-final placement, higher=better, range 1..4, avg-rank ties."""
    seat = seat.astype(np.int64)
    score = score.astype(np.float64)
    ug = np.unique(game)
    gidx = np.searchsorted(ug, game)
    ngames = len(ug)
    scores4 = np.full((ngames, 4), np.nan)
    scores4[gidx, seat] = score
    assert not np.isnan(scores4).any(), "some game is missing a seat"
    gt = (scores4[:, None, :] > scores4[:, :, None]).sum(2)      # n_greater
    eq = (scores4[:, None, :] == scores4[:, :, None]).sum(2)     # n_equal (incl self)
    place4 = 5.0 - (gt + (eq + 1) / 2.0)
    assert np.allclose(place4.sum(1), 10.0), "per-game placements must sum to 10"
    return place4[gidx, seat].astype(np.float32)


def predict_place(net, obs_arr, idx, dev):
    preds = np.empty(len(idx), np.float32)
    net.eval()
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]
            ob = torch.from_numpy(np.ascontiguousarray(obs_arr[b])).float().to(dev)
            with torch.cuda.amp.autocast():
                preds[i:i + 4096] = net(ob, None).float().cpu().numpy()
    return preds


class EMA:
    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in net.state_dict().items()}

    def update(self, net):
        d = self.decay
        for k, v in net.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, net):
        sd = net.state_dict()
        net.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="deploy ckpt path (.pt)")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))  # keep load<100
    dev = "cuda"
    t0 = time.time()

    d = np.load(F2_CORPUS)
    obs = d["obs"]
    game, seat = d["game"], d["seat"]
    y = compute_placement(game, seat, d["score"])
    stage = stage_of(d["step"], d["gamelen"])
    print(f"placement base-rate mean={y.mean():.4f} min={y.min()} max={y.max()} N={len(y):,}",
          flush=True)

    vidx, tidx = split_val_games(game)              # rng 777, 10% by game (v1 verbatim)
    vg, tg = set(game[vidx].tolist()), set(game[tidx].tolist())
    assert vg.isdisjoint(tg), "train/val games overlap!"
    print(f"GAME-DISJOINT ok: train_games={len(tg)} val_games={len(vg)} "
          f"train_rows={len(tidx):,} val_rows={len(vidx):,} split_seed=777", flush=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    net = VNet(cond=False).to(dev)
    ema = EMA(net, decay=0.999)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    scaler = torch.cuda.amp.GradScaler()
    rb = np.random.RandomState(2000 + a.seed)

    net.train()
    for s in range(a.steps):
        bi = tidx[np.sort(rb.randint(0, len(tidx), 512))]
        ob = torch.from_numpy(np.ascontiguousarray(obs[bi])).float().to(dev)
        yb = torch.from_numpy(y[bi]).to(dev)
        with torch.cuda.amp.autocast():
            loss = F.mse_loss(net(ob, None).float(), yb)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt)
        scaler.update(); sch.step(); ema.update(net)
        if s % 2000 == 0:
            print(f"  step {s}/{a.steps} mse {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

    # eval RAW net (learning signal / smoke gate)
    pr = predict_place(net, obs, vidx, dev)
    mr = metrics(pr, y[vidx], game[vidx], seat[vidx], stage[vidx])
    mr["val_mse"] = round(float(np.mean((pr - y[vidx]) ** 2)), 4)
    mr["mean_pred"] = round(float(pr.mean()), 4)
    print(f"RAW_R r_all={mr['r_all']} rho_all={mr['rho_all']} mean_pred={mr['mean_pred']}", flush=True)

    # eval with EMA weights (deploy)
    eval_net = VNet(cond=False).to(dev)
    ema.copy_to(eval_net)
    pv = predict_place(eval_net, obs, vidx, dev)
    m = metrics(pv, y[vidx], game[vidx], seat[vidx], stage[vidx])
    m["val_mse"] = round(float(np.mean((pv - y[vidx]) ** 2)), 4)
    m["mean_pred"] = round(float(pv.mean()), 4)
    m["mean_true"] = round(float(y[vidx].mean()), 4)
    m["raw_r_all"] = mr["r_all"]
    m["raw_mean_pred"] = mr["mean_pred"]
    print("VAL " + json.dumps(m), flush=True)
    print(f"VAL_R r_all={m['r_all']} rho_all={m['rho_all']} mean_pred={m['mean_pred']} "
          f"(raw r_all={mr['r_all']} mean_pred={mr['mean_pred']})", flush=True)

    # save deployable EMA checkpoint (PIMC leaf evaluator)
    ck = a.ckpt or (a.out[:-5] + ".pt" if a.out.endswith(".json") else a.out + ".pt")
    os.makedirs(os.path.dirname(ck) or ".", exist_ok=True)
    torch.save(eval_net.state_dict(), ck)
    print("SAVED_CKPT " + ck, flush=True)

    out = dict(mode="placeval_resbn128x40_ema", seed=a.seed, steps=a.steps,
               target="game_final_placement_higher_better_1to4", loss="mse",
               conditioning=False, ema_decay=0.999, corpus=F2_CORPUS,
               split_seed=777, n_train_games=len(tg), n_val_games=len(vg),
               n_train_rows=int(len(tidx)), n_val_rows=int(len(vidx)),
               placement_base_rate=round(float(y.mean()), 4),
               ckpt=ck, metrics=m, seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("DONE " + a.out, flush=True)


if __name__ == "__main__":
    main()
