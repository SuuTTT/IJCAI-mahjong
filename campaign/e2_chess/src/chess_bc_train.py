#!/usr/bin/env python3
"""E2 chess BC teacher: ResNet c64x6 (throughput_probe arch), CE on (obs -> played move).

Data: sharded npz from encode_games.py. Val split is BY GAME with a fixed split
seed (1234) shared by every model of a band, so val positions are identical
across teachers/students of that band; the model --seed controls init + shuffle
order only. Checkpoint is written atomically (tmp -> os.replace) so the GPU
keeper's REQUIRE file-count gates only ever see complete checkpoints.
"""
import argparse, glob, json, os, sys, time
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from throughput_probe import resnet  # single source of truth for the arch

SPLIT_SEED = 1234
VAL_FRAC = 0.02


def load_band(enc_dir):
    files = sorted(glob.glob(os.path.join(enc_dir, "shard_*.npz")))
    assert files, f"no shards in {enc_dir}"
    obs_l, act_l, gidx_l, goff = [], [], [], 0
    for f in files:
        z = np.load(f)
        obs_l.append(z["obs"])
        act_l.append(z["action"].astype(np.int64))
        gidx_l.append(z["game_idx"].astype(np.int64) + goff)
        goff += len(z["game_id"])
    return (np.concatenate(obs_l), np.concatenate(act_l),
            np.concatenate(gidx_l), goff)


def game_split(gidx, n_games):
    rng = np.random.default_rng(SPLIT_SEED)
    val_game = rng.random(n_games) < VAL_FRAC
    val_mask = val_game[gidx]
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)


def evaluate(torch, net, obs_t, act_t, idx, dev, bs=4096):
    net.eval()
    ce, top1, n = 0.0, 0, 0
    lossf = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs])
            x = obs_t[j].to(dev).float()
            y = act_t[j].to(dev)
            logits = net(x)
            ce += lossf(logits, y).item()
            top1 += (logits.argmax(1) == y).sum().item()
            n += len(j)
    net.train()
    return ce / n, top1 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    obs, act, gidx, ng = load_band(a.enc_dir)
    tr, va = game_split(gidx, ng)
    print(json.dumps({"positions": int(len(act)), "games": int(ng),
                      "train": int(len(tr)), "val": int(len(va)),
                      "load_s": round(time.time() - t0, 1), "device": dev,
                      "seed": a.seed}), flush=True)
    obs_t = torch.from_numpy(obs)  # uint8, stays on CPU
    act_t = torch.from_numpy(act)
    del obs, act, gidx

    net = resnet(a.channels, a.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    steps_total = max(1, a.epochs * (len(tr) // a.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(a.seed)
    hist, step = [], 0
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        te0, run_loss, run_n = time.time(), 0.0, 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            j = torch.from_numpy(perm[i:i + a.bs])
            x = obs_t[j].to(dev).float()
            y = act_t[j].to(dev)
            loss = lossf(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run_loss += loss.item() * a.bs; run_n += a.bs; step += 1
            if step % 2000 == 0:
                print(json.dumps({"ep": ep, "step": step,
                                  "train_ce": round(run_loss / run_n, 4),
                                  "pos_per_s": round(run_n / (time.time() - te0), 0)}),
                      flush=True)
        vce, vacc = evaluate(torch, net, obs_t, act_t, va, dev)
        rec = {"ep": ep, "train_ce": round(run_loss / max(run_n, 1), 4),
               "val_ce": round(vce, 4), "val_top1": round(vacc, 4),
               "ep_s": round(time.time() - te0, 1)}
        hist.append(rec); print(json.dumps(rec), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    torch.save({"state_dict": net.state_dict(), "channels": a.channels,
                "blocks": a.blocks, "seed": a.seed, "enc_dir": a.enc_dir,
                "epochs": a.epochs, "bs": a.bs, "lr": a.lr,
                "kind": "teacher", "hist": hist}, tmp)
    os.replace(tmp, a.out)
    with open(a.out + ".traininfo.json", "w") as f:
        json.dump({"out": a.out, "kind": "teacher", "seed": a.seed,
                   "positions": int(len(act_t)), "games": int(ng),
                   "hist": hist}, f, indent=2)
    print("SAVED " + a.out, flush=True)


if __name__ == "__main__":
    main()
