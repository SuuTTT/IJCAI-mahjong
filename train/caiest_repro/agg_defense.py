"""agg_defense.py -- combine null-cal + tau/K cells into results/DEFENSE_GATE.json."""
import json, glob, os
R = "/root/caiest_repro/results"

def load(p):
    try: return json.load(open(p))
    except Exception: return None

null_short = load(f"{R}/def_nullcal.json")
null_tau1 = load(f"{R}/def_tau1cal.json")
cells = []
order = ["tau0.3_K3", "tau0.5_K3", "tau0.7_K3", "tau0.0_K3", "tau0.0_Kall"]
for name in order:
    d = load(f"{R}/DEFENSE_cell_{name}.json")
    if d is None: continue
    cells.append(dict(name=name, tau=d["tau"], K=d["K"],
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
    overall = ("DEPLOYABLE DEFENSE BEATS kdens3 -- configs with CI>2.5: "
               + ", ".join(f"{c['name']}({c['mean_placement']:.4f})" for c in beaters))
else:
    overall = ("NO defensive config beats kdens3 (no CI lower-bound > 2.500). "
               "Great deal-in AUROC (0.97) did NOT translate to better placement -- "
               "danger prediction != better placement (the campaign's recurring lesson).")

out = dict(
    experiment="deployable defensive discard policy (per-candidate deal-in, top-K safe divert) vs kdens3",
    kd=["ckpt/kd/kd_128x40_s%d.pkl" % i for i in range(3)],
    pc=["ckpt/dealin_pc/dealin_pc_s%d.pt" % i for i in range(3)],
    null_cal=dict(
        shortcut_null=(null_short["block_mean_placement"] if null_short else None),
        tau1_fullpath=(null_tau1["block_mean_placement"] if null_tau1 else None),
        tau1_overrides=(null_tau1["n_overrides"] if null_tau1 else None),
        tau1_decisions_evaluated=(null_tau1["n_discard_decisions"] if null_tau1 else None),
        ok=bool(null_short and abs(null_short["block_mean_placement"] - 2.5) < 1e-9
                and null_tau1 and abs(null_tau1["block_mean_placement"] - 2.5) < 1e-9
                and null_tau1["n_overrides"] == 0)),
    gate="paired duplicate placement vs kdens3; 4 blocks x 500 seeds x 4 seat-rotations; "
         "seed0=9_000_000+block*3000 (disjoint walls); champion null = 2.5000",
    cells=cells,
    best_config=(dict(name=best["name"], mean_placement=best["mean_placement"],
                      ci95=best["ci95"]) if best else None),
    verdict=overall)
os.makedirs(R, exist_ok=True)
json.dump(out, open(f"{R}/DEFENSE_GATE.json", "w"), indent=1)
print(json.dumps(out, indent=1))
