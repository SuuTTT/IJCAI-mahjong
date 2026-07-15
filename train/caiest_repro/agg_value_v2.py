"""agg_value_v2.py — aggregate value_v2_{a,b,c}_s{0,1}.json + v1 (value_e2e.json) into
results/VALUE_V2.json with the comparison table and a factual verdict."""
import os, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KEYS = ["r_all", "rho_all", "r_early", "r_mid", "r_late", "rho_early", "rho_mid", "rho_late",
        "grp_rank_rho_early", "grp_rank_rho_mid", "grp_rank_rho_late", "val_mse"]

runs = []
for f in sorted(glob.glob(os.path.join(RES, "value_v2_[abc]_s[01].json"))):
    runs.append(json.load(open(f)))

v1 = json.load(open(os.path.join(RES, "value_e2e.json")))

table = {}
for v in ["a", "b", "c"]:
    rs = [r for r in runs if r["variant"] == v]
    if not rs:
        continue
    ent = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs]}
    for dom in ["final2", "official"]:
        agg = {}
        for k in KEYS:
            vals = [r[f"metrics_{dom}"][k] for r in rs if r[f"metrics_{dom}"].get(k) is not None]
            if vals:
                agg[k] = round(float(np.mean(vals)), 4)
                agg[k + "_per_seed"] = vals
        ent[dom] = agg
    table[v] = ent

def g(v, dom, k):
    return table.get(v, {}).get(dom, {}).get(k)

verdict = {}
if "a" in table and "b" in table and "c" in table:
    v1r, v1late = v1["metrics"]["r_all"], v1["metrics"]["r_late"]
    best = max(["b", "c"], key=lambda v: g(v, "final2", "r_all"))
    verdict = {
        "v1_final2": {"r_all": v1r, "r_late": v1late},
        "a_repro_final2": {"r_all": g("a", "final2", "r_all"), "r_late": g("a", "final2", "r_late")},
        "best_v2_variant_on_final2": best,
        "best_v2_final2": {"r_all": g(best, "final2", "r_all"), "r_late": g(best, "final2", "r_late")},
        "more_data_helps_final2_b_minus_a_r_all": round(g("b", "final2", "r_all") - g("a", "final2", "r_all"), 4),
        "conditioning_helps_final2_c_minus_b_r_all": round(g("c", "final2", "r_all") - g("b", "final2", "r_all"), 4),
        "conditioning_helps_official_c_minus_b_r_all": round(g("c", "official", "r_all") - g("b", "official", "r_all"), 4),
        "cross_domain_a_on_official_r_all": g("a", "official", "r_all"),
        "beats_v1_r_all": bool(g(best, "final2", "r_all") > v1r),
        "beats_v1_r_late": bool(g(best, "final2", "r_late") > v1late),
    }

out = {
    "campaign": "VALUE-HEAD V2 (more data + source conditioning) vs v1 value_e2e",
    "protocol": "identical to v1: by-game split rng777/10%, SC=30, 30k steps e2e 128x40, "
                "r/rho overall+stage + GRP within-game rank-rho; official held-out evaluated "
                "for all variants (cross-domain for a)",
    "variants": {"a": "v1-repro final2-only", "b": "final2+official 50/50, no cond",
                 "c": "final2+official 50/50 + source embedding (bot 0-3, official=4)"},
    "v1": v1, "table": table, "runs": runs, "verdict": verdict,
}
with open(os.path.join(RES, "VALUE_V2.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({"table": {v: {d: {k: table[v][d].get(k) for k in ["r_all", "r_late"]}
                                for d in ["final2", "official"]} for v in table},
                  "verdict": verdict}, indent=2))
