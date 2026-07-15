"""Merge the fully-uniform-protocol curve: eps 0.1/0.2/0.5 from regate_eXX_v2.npz (already
sweep protocol: 11k-game data, 6 BC teachers, 6 KD students distilled from all 6) + fresh
protocol-matched eps 0/0.3 legs (regate_e00s/e03s_v3.npz). All gated on the SAME per-game-
seeded 3000-seed set (10000-12999, seat 0 vs 2 rule agents, paired).
Writes results/noise_sweep_v3.json, including movement vs the v2 (mismatched-protocol)
eps 0/0.3 points."""
import json, math
import numpy as np

SRC = {"0": "regate_e00s_v3", "0.1": "regate_e01_v2", "0.2": "regate_e02_v2",
       "0.3": "regate_e03s_v3", "0.5": "regate_e05_v2"}

def summarize(npz_path, json_path):
    meta = json.load(open(json_path))
    assert meta["done"] == meta["total"], f"{json_path} incomplete ({meta['done']}/{meta['total']})"
    arrs = dict(np.load(npz_path).items())
    disjoint = sorted(k for k in arrs if k.startswith("seed_trio") and "overlap" not in k)
    dists = sorted(k for k in arrs if k.startswith("distill"))
    se_means = {k: round(float(arrs[k].mean()), 5) for k in disjoint}
    de_means = {k: round(float(arrs[k].mean()), 5) for k in dists}
    se_avg = np.mean([arrs[k] for k in disjoint], axis=0)
    de_avg = np.mean([arrs[k] for k in dists], axis=0)
    gap = de_avg - se_avg
    gap_ci = float(1.96 * gap.std(ddof=1) / math.sqrt(len(gap)))
    pair_gaps = [round(float(arrs[d].mean() - arrs[s].mean()), 5) for d in dists for s in disjoint]
    return {"seed_ens_trios": se_means,
            "seed_ens_mean": round(float(se_avg.mean()), 5),
            "seed_ens_spread": round(max(se_means.values()) - min(se_means.values()), 5),
            "distill_ens": de_means,
            "distill_mean": round(float(de_avg.mean()), 5),
            "gap_mean": round(float(gap.mean()), 5),
            "gap_ci95_paired": round(gap_ci, 5),
            "gap_range": [min(pair_gaps), max(pair_gaps)],
            "pairwise_gaps": pair_gaps}

by = {}
for eps, tag in SRC.items():
    by[eps] = summarize(f"results/{tag}.npz", f"results/{tag}.json")

# movement of the re-trained eps 0 / 0.3 legs vs the v2 mismatched-protocol legs
v2 = json.load(open("results/noise_sweep_v2.json"))["by_eps"]
for eps in ("0", "0.3"):
    by[eps]["v2_mismatched_protocol"] = {
        "seed_ens_mean": v2[eps]["seed_ens_mean"], "distill_mean": v2[eps]["distill_mean"],
        "gap_mean": v2[eps]["gap_mean"],
        "delta_gap_v3_minus_v2": round(by[eps]["gap_mean"] - v2[eps]["gap_mean"], 5)}

res = {
    "experiment": "noise_level_epsilon_sweep_v3_uniform_protocol",
    "audit_fixes": {
        "C2": "seed-ensemble payoff averaged over 2 DISJOINT trios per eps (spread reported), "
              "not a single trio",
        "M1": "eps 0 and 0.3 legs RE-TRAINED under the exact sweep protocol (11k-game data, "
              "6 BC teachers 60k steps, 6 KD students 25k steps alpha .7 distilled from all 6) "
              "replacing the 8-teacher/larger-data legs"},
    "metric": "mean_payoff seat0 vs 2 DouDizhuRuleAgentV1",
    "gate": "per-game-seeded paired eval, seeds 10000-12999 (3000), identical across all "
            "ensembles and eps",
    "by_eps": {k: by[k] for k in ["0", "0.1", "0.2", "0.3", "0.5"]},
}
json.dump(res, open("results/noise_sweep_v3.json", "w"), indent=2)
print(json.dumps(res, indent=2))
