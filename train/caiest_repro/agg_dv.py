"""agg_dv.py -- combine null-cal + lambda/K cells into results/DEFENSE_VALUE_GATE.json."""
import json, os
R = "/root/caiest_repro/results"

def load(p):
    try: return json.load(open(p))
    except Exception: return None

null = load(f"{R}/dv_nullcal.json")
order = ["lam0_K3", "lam0.5_K3", "lam1_K3", "lam2_K3", "lam4_K3", "lam0_Kall"]
# include any best-lambda K=all cells added later
import glob
extra = [os.path.basename(f)[8:-5] for f in glob.glob(f"{R}/DV_cell_*.json")]
for e in extra:
    if e not in order: order.append(e)
cells = []
for name in order:
    d = load(f"{R}/DV_cell_{name}.json")
    if d is None: continue
    cells.append(dict(name=name, lam=d["lam"], K=d["K"],
                      block_means=d["block_means"], mean_placement=d["block_mean_placement"],
                      ci95=d["ci95"], ci95_halfwidth=d["ci95_halfwidth"], block_sd=d["block_sd"],
                      override_fraction=d["override_fraction"], n_overrides=d["n_overrides"],
                      n_discard_decisions=d["n_discard_decisions"], verdict=d["verdict"],
                      seconds=d.get("seconds")))

beaters = [c for c in cells if c["ci95"][0] > 2.5]
best = max(cells, key=lambda c: c["mean_placement"]) if cells else None
if not cells:
    overall = "NO CELLS YET"
elif beaters:
    overall = ("DEPLOYABLE WINNER: value-aware action-value defense BEATS kdens3 (CI>2.5): "
               + ", ".join(f"{c['name']}({c['mean_placement']:.4f})" for c in beaters)
               + " -- FIRST deployable model to beat the champion.")
else:
    overall = ("NO lambda/K config beats kdens3 (no CI lower-bound > 2.500). The static 1-ply "
               "joint action-value (offense value - lambda*P_dealin*L) cannot capture the oracle "
               "rollout's 3.55 -> the win needs true LOOKAHEAD / PIMC, not a static heuristic. "
               "Build PIMC.")

out = dict(
    experiment="value-aware action-value defense A=V-lam*P_dealin*L vs kdens3",
    formula="A(T) = mean_5_valueheads(V post-discard, src=0) - lambda * mean_3_pc(P_dealin) * L",
    L_dealin=(load(f"{R}/DV_cell_lam1_K3.json") or {}).get("L_dealin", 0.6834),
    SC=30.0, avg_fan=12.5, deploy_src=0,
    value_heads=["results/VALUE_C_60K.pt", "results/VALUE_C_60K_s1.pt", "results/VALUE_C_60K_s3.pt",
                 "results/VALUE_C_60K_s4.pt", "results/VALUE_C_60K_s6.pt"],
    pc=["ckpt/dealin_pc/dealin_pc_s%d.pt" % i for i in range(3)],
    kd=["ckpt/kd/kd_128x40_s%d.pkl" % i for i in range(3)],
    null_cal=dict(
        block_mean=(null["block_mean_placement"] if null else None),
        overrides=(null["n_overrides"] if null else None),
        decisions_evaluated=(null["n_discard_decisions"] if null else None),
        ok=bool(null and abs(null["block_mean_placement"] - 2.5) < 1e-9 and null["n_overrides"] == 0)),
    reference=dict(kdens3_null=2.5, stage1_offense_valuemax_fulllegal=2.026,
                   naive_defense_override_best=2.4995, oracle_jointEV_rollout=3.55),
    gate="paired duplicate placement vs kdens3; 4 blocks x 500 seeds x 4 rotations; "
         "seed0=9_500_000+block*3000 (disjoint walls); champion null=2.5000",
    cells=cells,
    best_config=(dict(name=best["name"], mean_placement=best["mean_placement"],
                      ci95=best["ci95"]) if best else None),
    verdict=overall,
    conclusion=("DEPLOYABLE-WINNER" if beaters else "NEED-PIMC (static 1-ply joint eval insufficient)"))
os.makedirs(R, exist_ok=True)
json.dump(out, open(f"{R}/DEFENSE_VALUE_GATE.json", "w"), indent=1)
print(json.dumps(out, indent=1))
