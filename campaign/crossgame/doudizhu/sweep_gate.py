"""Per-eps sweep gate over ONE fixed per-game-seeded seed set: single / 3-teacher seed-ensemble /
3-student distill-ensemble (group A), plus an independent group-B distill-ensemble, and a
per-game PAIRED gap CI (distill_A - seed_ens over all games). ref==ref calibration included."""
import os, sys, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from dou_gate import EnsembleAgent
from gate_curve import play_perseed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)              # eps value as string
    ap.add_argument("--single", required=True)
    ap.add_argument("--seed_teachers", required=True)    # comma 3
    ap.add_argument("--distillA", required=True)         # comma 3
    ap.add_argument("--distillB", required=True)         # comma 3
    ap.add_argument("--nseeds", type=int, default=3000)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(a.seed0, a.seed0 + a.nseeds))
    def play(pkls):
        return play_perseed(EnsembleAgent(pkls, a.hidden, a.layers, dev), a.seat, seeds)

    sm = play([a.single])
    se = play(a.seed_teachers.split(","))
    da = play(a.distillA.split(","))
    db = play(a.distillB.split(","))
    ref = play([a.single])                                # calibration copy
    gap_arr = da - se                                     # per-game paired gap (group A)
    mean_gap = float(gap_arr.mean())
    gap_ci = float(1.96 * gap_arr.std(ddof=1) / math.sqrt(len(gap_arr)))
    res = {"eps": float(a.tag), "nseeds": a.nseeds, "seed0": a.seed0, "seat": a.seat, "dev": dev,
           "single": round(float(sm.mean()), 5),
           "seed_ens": round(float(se.mean()), 5),
           "distill_ens": round(float(da.mean()), 5),       # group A = primary
           "distill_ensB": round(float(db.mean()), 5),      # group B = independent replicate
           "gap": round(mean_gap, 5),                        # distill_ens - seed_ens (paired)
           "gap_ci95": round(gap_ci, 5),
           "gap_B": round(float(db.mean() - se.mean()), 5),
           "calibration_delta": round(abs(float(sm.mean() - ref.mean())), 6)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res), flush=True)
    print(f"WROTE {a.out}", flush=True)

if __name__ == "__main__":
    main()
