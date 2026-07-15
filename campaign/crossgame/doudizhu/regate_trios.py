"""C2 audit fix: re-gate seed-ensembles over MULTIPLE trios per eps (plus distill
ensembles) over ONE fixed per-game-seeded seed set (paired, same protocol as sweep_gate.py:
play_perseed, seat 0 vs 2 DouDizhuRuleAgentV1, seeds seed0..seed0+nseeds-1).
Saves per-game payoff arrays (npz) incrementally + a means json, so paired gaps and
trio spread can be computed exactly."""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from dou_gate import EnsembleAgent
from gate_curve import play_perseed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)                 # eps value as string
    ap.add_argument("--ens", action="append", required=True,
                    help="label=pkl1,pkl2,... (repeatable)")
    ap.add_argument("--nseeds", type=int, default=3000)
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(a.seed0, a.seed0 + a.nseeds))
    ens_map = {}
    for spec in a.ens:
        label, pkls = spec.split("=", 1)
        ens_map[label] = pkls.split(",")
    arrays, means = {}, {}
    for label, pkls in ens_map.items():
        ag = EnsembleAgent(pkls, a.hidden, a.layers, dev)
        v = play_perseed(ag, a.seat, seeds)
        arrays[label] = v
        means[label] = round(float(v.mean()), 5)
        print(f"[eps={a.tag}] {label}: mean_payoff={means[label]}", flush=True)
        # incremental save so partial results survive interruption
        np.savez(a.npz, **arrays)
        with open(a.out, "w") as f:
            json.dump({"eps": a.tag, "nseeds": a.nseeds, "seed0": a.seed0,
                       "seat": a.seat, "dev": dev, "means": means,
                       "ens": ens_map,
                       "done": len(arrays), "total": len(ens_map)}, f, indent=2)
    print(f"DONE eps={a.tag} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
