"""
danger_std_check.py — JD-v2 step 1: VERIFY the constant-penalty hypothesis.

Computes, over 10k random cooked states (with >=2 legal discards), the per-state std of the
danger head's 34 discard sigmoids: over ALL 34 slots and over the LEGAL discards only.
If that std is ~0 within states, v1's pen = E[p*danger] had ~no policy gradient.
-> results/DANGER_STD.json
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from models_explore import ResBNCNN

HERE = os.path.dirname(os.path.abspath(__file__))
PLAY0, NPLAY = 2, 34


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--danger", default="ckpt/danger/danger4.bn.pkl")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", default="results/DANGER_STD.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(os.path.join(HERE, "data", "cooked_single.npz"))
    o, m = d["obs"], d["mask"]
    rng = np.random.RandomState(123)
    # states with >=2 legal discards (where the penalty could differentiate)
    nleg_play = m[:, PLAY0:PLAY0 + NPLAY].sum(1)
    cand = np.flatnonzero(nleg_play >= 2)
    idx = np.sort(cand[rng.permutation(len(cand))[:a.n]])
    net = ResBNCNN(channels=128, blocks=40)
    net.load_state_dict(torch.load(a.danger, map_location="cpu")); net.eval().to(dev)
    stds_all, stds_leg, rng_leg, means_leg = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]
            ob = torch.from_numpy(np.ascontiguousarray(o[b])).float().to(dev)
            mk = torch.from_numpy(np.ascontiguousarray(m[b])).to(dev)
            lg = net({"is_training": False,
                      "obs": {"observation": ob,
                              "action_mask": torch.ones(ob.shape[0], 235, device=dev)}})
            dg = torch.sigmoid(lg[:, PLAY0:PLAY0 + NPLAY].float())
            stds_all.append(dg.std(dim=1).cpu().numpy())
            lp = mk[:, PLAY0:PLAY0 + NPLAY].float()
            nl = lp.sum(1).clamp(min=1.0)
            mu = (dg * lp).sum(1) / nl
            var = ((dg - mu[:, None]) ** 2 * lp).sum(1) / nl
            stds_leg.append(var.clamp(min=0).sqrt().cpu().numpy())
            big = torch.where(lp > 0, dg, torch.full_like(dg, -1.0)).max(1).values
            small = torch.where(lp > 0, dg, torch.full_like(dg, 2.0)).min(1).values
            rng_leg.append((big - small).cpu().numpy())
            means_leg.append(mu.cpu().numpy())
    sa = np.concatenate(stds_all); sl = np.concatenate(stds_leg)
    rl = np.concatenate(rng_leg); ml = np.concatenate(means_leg)
    out = dict(
        n_states=int(len(idx)), danger=a.danger,
        std_all34=dict(mean=round(float(sa.mean()), 5), median=round(float(np.median(sa)), 5),
                       p10=round(float(np.percentile(sa, 10)), 5),
                       p90=round(float(np.percentile(sa, 90)), 5)),
        std_legal=dict(mean=round(float(sl.mean()), 5), median=round(float(np.median(sl)), 5),
                       p10=round(float(np.percentile(sl, 10)), 5),
                       p90=round(float(np.percentile(sl, 90)), 5)),
        range_legal_mean=round(float(rl.mean()), 5),
        mean_danger_legal=round(float(ml.mean()), 5),
        between_state_std_of_mean=round(float(ml.std()), 5),
        note="v1 penalty gradient ~ within-state spread (std_legal); "
             "constant component = mean_danger_legal (removed in v2)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
