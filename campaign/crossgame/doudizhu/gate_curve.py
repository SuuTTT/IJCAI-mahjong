"""Gate one ensemble candidate over a fixed seed set; report mean payoff + 95% CI (per-seed).
Optionally a single-net calibrated reference over the SAME seeds. JSON out."""
import os, sys, argparse, json, math, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import rlcard
from rlcard.models.doudizhu_rule_models import DouDizhuRuleAgentV1
from dou_gate import EnsembleAgent

def play_perseed(cand, seat, seeds):
    rule = DouDizhuRuleAgentV1()
    vals = []
    for sd in seeds:
        np.random.seed(int(sd)); random.seed(int(sd))  # reproducible rule-agent opponents per seed
        env = rlcard.make("doudizhu", config={"seed": int(sd)})
        agents = [rule, rule, rule]; agents[seat] = cand
        env.set_agents(agents)
        _, payoffs = env.run(is_training=False)
        vals.append(float(payoffs[seat]))
    return np.array(vals)

def summ(v):
    m = float(v.mean()); ci = float(1.96 * v.std(ddof=1) / math.sqrt(len(v)))
    return round(m, 5), round(ci, 5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls", required=True, help="comma ensemble pkls")
    ap.add_argument("--ref", default="", help="single reference pkl (calibration)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--nseeds", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(20000, 20000 + a.nseeds))
    t0 = time.time()
    cand = EnsembleAgent(a.pkls.split(","), a.hidden, a.layers, dev)
    v = play_perseed(cand, a.seat, seeds); m, ci = summ(v)
    res = {"tag": a.tag, "pkls": a.pkls.split(","), "K": len(a.pkls.split(",")),
           "seat": a.seat, "nseeds": a.nseeds, "mean_payoff": m, "ci95": ci, "dev": dev,
           "seconds": round(time.time() - t0, 1)}
    if a.ref:
        rv = play_perseed(EnsembleAgent([a.ref], a.hidden, a.layers, dev), a.seat, seeds)
        rm, rci = summ(rv)
        res["ref_mean_payoff"] = rm; res["ref_ci95"] = rci
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
