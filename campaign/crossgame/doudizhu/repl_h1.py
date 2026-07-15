"""Noisy-H1 replication over a POOLED set of independent students (all distilled from the 8 noisy
teachers s200-207). Forms disjoint 3-student groups -> distill-ensembles; each block gated vs a
3-teacher seed-ensemble (rotating window) over a FRESH per-block 2000-seed set (per-game-seeded).
Aggregate -> {n_blocks, mean_distill, mean_seed, mean_gap, gap_ci95_lo, gap_ci95_hi}."""
import os, sys, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from dou_gate import EnsembleAgent
from gate_curve import play_perseed

T95 = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", required=True, help="comma-separated POOL of student pkls")
    ap.add_argument("--teachers", required=True, help="comma-separated 8 noisy teacher pkls")
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--nseeds", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--out", default="results/noisy_h1_repl.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    students = [s for s in a.students.split(",") if s]
    teachers = [t for t in a.teachers.split(",") if t]
    nb = len(students) // a.group
    print(f"pool={len(students)} students -> {nb} disjoint groups of {a.group}; {len(teachers)} teachers", flush=True)

    blocks = []
    for i in range(nb):
        distill = students[i*a.group:(i+1)*a.group]
        tri = [teachers[(i + j) % len(teachers)] for j in range(3)]
        s0 = a.seed0 + i * a.nseeds
        seeds = list(range(s0, s0 + a.nseeds))
        dv = play_perseed(EnsembleAgent(distill, a.hidden, a.layers, dev), a.seat, seeds)
        sv = play_perseed(EnsembleAgent(tri, a.hidden, a.layers, dev), a.seat, seeds)
        dmean, smean = float(dv.mean()), float(sv.mean())
        blocks.append({"block": i, "seed0": s0,
                       "distill_students": [os.path.basename(p) for p in distill],
                       "seed_teachers": [os.path.basename(p) for p in tri],
                       "distill_ens_payoff": round(dmean, 5), "seed_ens_payoff": round(smean, 5),
                       "gap": round(dmean - smean, 5)})
        print(f"block {i}: distill={dmean:.5f} seed={smean:.5f} gap={dmean-smean:+.5f}", flush=True)

    g = np.array([b["gap"] for b in blocks])
    d = np.array([b["distill_ens_payoff"] for b in blocks])
    s = np.array([b["seed_ens_payoff"] for b in blocks])
    mean_gap = float(g.mean())
    dof = max(1, len(g) - 1)
    se = float(g.std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else float("nan")
    t = T95.get(dof, 1.96)
    out = {"regime": "noisy_eps0.3", "metric": "mean_payoff seat0 vs 2 rule agents",
           "n_blocks": nb, "nseeds_per_block": a.nseeds,
           "mean_distill": round(float(d.mean()), 5), "mean_seed": round(float(s.mean()), 5),
           "mean_gap": round(mean_gap, 5),
           "gap_sd": round(float(g.std(ddof=1)), 5) if len(g) > 1 else None,
           "gap_ci95_lo": round(mean_gap - t * se, 5) if len(g) > 1 else None,
           "gap_ci95_hi": round(mean_gap + t * se, 5) if len(g) > 1 else None,
           "blocks": blocks}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("n_blocks", "mean_distill", "mean_seed",
          "mean_gap", "gap_ci95_lo", "gap_ci95_hi")}), flush=True)
    print(f"WROTE {a.out}", flush=True)

if __name__ == "__main__":
    main()
