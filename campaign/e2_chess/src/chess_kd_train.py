#!/usr/bin/env python3
"""E2 chess KD student: distill from the mean softmax of the band's teachers.

loss = alpha * CE(student, mean_k softmax(teacher_k(x))) + (1-alpha) * CE(student, played move)
alpha = 0.7, T = 1 (same 0.7 soft + 0.3 hard convention as the mahjong/CIFAR KD recipe).
Soft targets are computed on the fly (6 frozen teachers, no_grad, same GPU).
Same data + game-level val split (split seed 1234) as chess_bc_train.py.
Checkpoint written atomically for the keeper's REQUIRE gates.
"""
import argparse, json, os, sys, time
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from throughput_probe import resnet
from chess_bc_train import load_band, game_split, evaluate


def load_frozen(torch, path, dev):
    ck = torch.load(path, map_location=dev)
    net = resnet(ck["channels"], ck["blocks"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-dir", required=True)
    ap.add_argument("--teachers", required=True, help="comma-separated ckpt paths")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    torch.manual_seed(a.seed)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tpaths = [p for p in a.teachers.split(",") if p]
    teachers = [load_frozen(torch, p, dev) for p in tpaths]

    t0 = time.time()
    obs, act, gidx, ng = load_band(a.enc_dir)
    tr, va = game_split(gidx, ng)
    print(json.dumps({"positions": int(len(act)), "games": int(ng),
                      "train": int(len(tr)), "val": int(len(va)),
                      "teachers": len(teachers), "alpha": a.alpha,
                      "load_s": round(time.time() - t0, 1), "device": dev,
                      "seed": a.seed}), flush=True)
    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(act)
    del obs, act, gidx

    net = resnet(a.channels, a.blocks).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    steps_total = max(1, a.epochs * (len(tr) // a.bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)
    rng = np.random.default_rng(a.seed)
    hist, step = [], 0
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        te0, run_loss, run_kd, run_ce, run_n = time.time(), 0.0, 0.0, 0.0, 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            j = torch.from_numpy(perm[i:i + a.bs])
            x = obs_t[j].to(dev).float()
            y = act_t[j].to(dev)
            with torch.no_grad():
                soft = torch.zeros(len(j), 4672, device=dev)
                for t in teachers:
                    soft += F.softmax(t(x), 1)
                soft /= len(teachers)
            logits = net(x)
            logp = F.log_softmax(logits, 1)
            kd = -(soft * logp).sum(1).mean()
            ce = F.cross_entropy(logits, y)
            loss = a.alpha * kd + (1.0 - a.alpha) * ce
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run_loss += loss.item() * a.bs; run_kd += kd.item() * a.bs
            run_ce += ce.item() * a.bs; run_n += a.bs; step += 1
            if step % 2000 == 0:
                print(json.dumps({"ep": ep, "step": step,
                                  "loss": round(run_loss / run_n, 4),
                                  "kd": round(run_kd / run_n, 4),
                                  "ce": round(run_ce / run_n, 4),
                                  "pos_per_s": round(run_n / (time.time() - te0), 0)}),
                      flush=True)
        vce, vacc = evaluate(torch, net, obs_t, act_t, va, dev)
        rec = {"ep": ep, "loss": round(run_loss / max(run_n, 1), 4),
               "kd": round(run_kd / max(run_n, 1), 4),
               "ce": round(run_ce / max(run_n, 1), 4),
               "val_ce": round(vce, 4), "val_top1": round(vacc, 4),
               "ep_s": round(time.time() - te0, 1)}
        hist.append(rec); print(json.dumps(rec), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    torch.save({"state_dict": net.state_dict(), "channels": a.channels,
                "blocks": a.blocks, "seed": a.seed, "enc_dir": a.enc_dir,
                "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "alpha": a.alpha,
                "kind": "student", "teachers": tpaths, "hist": hist}, tmp)
    os.replace(tmp, a.out)
    with open(a.out + ".traininfo.json", "w") as f:
        json.dump({"out": a.out, "kind": "student", "seed": a.seed,
                   "alpha": a.alpha, "teachers": tpaths,
                   "positions": int(len(act_t)), "games": int(ng),
                   "hist": hist}, f, indent=2)
    print("SAVED " + a.out, flush=True)


if __name__ == "__main__":
    main()
