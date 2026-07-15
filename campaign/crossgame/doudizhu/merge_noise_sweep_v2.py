"""Merge per-eps trio re-gates (regate_eXX_v2.{json,npz}) into results/noise_sweep_v2.json.
C2 fix: seed-ensemble payoff = average over MULTIPLE trios (disjoint trios primary), with
trio spread reported; gap = distill_mean - seed_ens_mean with a per-game PAIRED CI computed
from the saved payoff arrays (all on the same 3000-seed set per eps)."""
import json, math
import numpy as np

FILES = {"0": "e00", "0.1": "e01", "0.2": "e02", "0.3": "e03", "0.5": "e05"}
PROTO = {
    "0": "8-teacher pool (dou_data.npz, det); distill students s100-102 distilled from ALL 8 "
         "teachers; original det_gate used 2000 seeds@10000 via dou_gate.play - re-gated here "
         "on the standard 3000-seed play_perseed protocol. Only ONE distill trio exists. "
         "trio3 overlaps trio1 (reuses teacher s0).",
    "0.3": "8-teacher pool (dou_data_noisy.npz); distill students s210-212 distilled from ALL 8 "
           "teachers; original noisy_gate used 2000 seeds@10000 via dou_gate.play - re-gated "
           "here on the standard 3000-seed play_perseed protocol. Only ONE distill trio exists. "
           "trio3 overlaps trio1 (reuses teacher s200).",
}

by = {}
for eps, tag in FILES.items():
    meta = json.load(open(f"results/regate_{tag}_v2.json"))
    arrs = dict(np.load(f"results/regate_{tag}_v2.npz").items())
    assert meta["done"] == meta["total"], f"eps={eps} incomplete ({meta['done']}/{meta['total']})"
    seed_labels = [k for k in arrs if k.startswith("seed_trio")]
    disjoint = [k for k in seed_labels if "overlap" not in k]
    dist_labels = [k for k in arrs if k.startswith("distill")]
    se_means = {k: round(float(arrs[k].mean()), 5) for k in seed_labels}
    de_means = {k: round(float(arrs[k].mean()), 5) for k in dist_labels}
    # trio-averaged per-game payoffs (paired across everything: same seeds)
    se_avg = np.mean([arrs[k] for k in disjoint], axis=0)
    de_avg = np.mean([arrs[k] for k in dist_labels], axis=0)
    gap = de_avg - se_avg
    n = len(gap)
    gap_mean = float(gap.mean())
    gap_ci = float(1.96 * gap.std(ddof=1) / math.sqrt(n))
    pair_gaps = [round(float(arrs[d].mean() - arrs[s].mean()), 5)
                 for d in dist_labels for s in disjoint]
    entry = {
        "seed_ens_trios": se_means,
        "seed_ens_mean": round(float(se_avg.mean()), 5),          # avg over DISJOINT trios
        "seed_ens_spread": round(max(se_means.values()) - min(se_means.values()), 5),
        "n_disjoint_trios": len(disjoint),
        "distill_ens": de_means,
        "distill_mean": round(float(de_avg.mean()), 5),
        "gap_mean": round(gap_mean, 5),
        "gap_ci95_paired": round(gap_ci, 5),
        "gap_range": [min(pair_gaps), max(pair_gaps)],
        "pairwise_gaps_distill_x_disjoint_trio": pair_gaps,
    }
    if eps in PROTO:
        entry["protocol_note"] = PROTO[eps]
    by[eps] = entry

res = {
    "experiment": "noise_level_epsilon_sweep_v2_trio_averaged",
    "audit_fix": "C2: v1 seed-ensemble used a SINGLE trio per eps; trio-to-trio variance is "
                 "comparable to the claimed effect. v2 averages over disjoint trios and reports "
                 "spread. All ensembles per eps gated on ONE fixed per-game-seeded 3000-seed "
                 "set (seeds 10000-12999, seat 0 vs 2 DouDizhuRuleAgentV1, paired).",
    "metric": "mean_payoff seat0 vs 2 rule agents",
    "seed_ens_mean_def": "mean payoff of trio-averaged per-game payoffs over disjoint trios "
                         "(overlapping trio3 shown in seed_ens_trios/spread but excluded from mean)",
    "by_eps": {k: by[k] for k in ["0", "0.1", "0.2", "0.3", "0.5"]},
}
json.dump(res, open("results/noise_sweep_v2.json", "w"), indent=2)
print(json.dumps(res, indent=2))
