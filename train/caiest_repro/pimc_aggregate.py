"""pimc_aggregate.py — pool per-block PIMC gate JSONs into per-tier mean/CI vs aug_s0 (2.500).
Ship rule: placement 95% CI lower bound > 2.500 (CI-separated better). Reads ONLY saved JSON."""
import os, sys, json, glob, math
import numpy as np

JDIR = sys.argv[1] if len(sys.argv) > 1 else "pimc_json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "SEARCH_RESULTS.json"

# group block files by tier tag "N{N}_H{H}"
tiers = {}
for fp in sorted(glob.glob(os.path.join(JDIR, "*.json"))):
    d = json.load(open(fp))
    tag = "N%d_H%d" % (d["N"], d["H"])
    tiers.setdefault(tag, []).append(d)

rows = []
for tag, blocks in sorted(tiers.items()):
    pl = np.array([b["placement_pts"] for b in blocks], dtype=np.float64)
    n = len(pl)
    mean = float(pl.mean())
    sd = float(pl.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    # 95% CI (t-ish; use 1.96 for n>=... we report both mean and normal CI)
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    margin_lo = ci_lo - 2.5
    tot_games = sum(b["games"] for b in blocks)
    tot_moves = sum(b["search_moves"] for b in blocks)
    tot_guard = sum(b.get("guarded", 0) for b in blocks)
    # per-move ms weighted by search_moves
    ms = sum(b["per_move_ms"] * b["search_moves"] for b in blocks) / tot_moves if tot_moves else 0.0
    ovr = sum(b["override_rate"] * b["search_moves"] for b in blocks) / tot_moves if tot_moves else 0.0
    beats = bool(ci_lo > 2.5)
    worse = bool(ci_hi < 2.5)
    rows.append(dict(
        tier=tag, N=blocks[0]["N"], H=blocks[0]["H"],
        topk=blocks[0]["topk"], delta=blocks[0]["delta"], margin=blocks[0]["margin"],
        n_blocks=n, total_games=tot_games,
        placement_mean=round(mean, 4), placement_sd=round(sd, 4), placement_se=round(se, 4),
        ci95_lo=round(ci_lo, 4), ci95_hi=round(ci_hi, 4), margin_lo=round(margin_lo, 4),
        beats_augs0=beats, worse_than_augs0=worse,
        per_move_ms=round(ms, 1), fits_6000ms=bool(ms <= 6000),
        override_rate=round(ovr, 4),
        search_moves=tot_moves, guarded_moves=tot_guard,
        first_pct=round(float(np.mean([b["first_pct"] for b in blocks])), 2),
        fourth_pct=round(float(np.mean([b["fourth_pct"] for b in blocks])), 2),
        block_placements=[round(x, 4) for x in pl.tolist()],
    ))

winners = [r for r in rows if r["beats_augs0"]]
best = None
if winners:
    best = max(winners, key=lambda r: r["margin_lo"])["tier"]

out = dict(
    experiment="PIMC / determinized value-search test-time lookahead vs aug_s0",
    base="aug_128x40_s0.pkl (current best; beats bn128s1 by +0.0058)",
    value_model="value_256x40.pkl (ValueMT 256x40, held-out 4th-AUC 0.955)",
    format="calibrated duplicate placement gate; aug_s0-vs-aug_s0 = 2.500 (verified N=0)",
    objective="minimize E[placement]: terminal Hu -> exact avg_rank; truncated -> V_place leaf (both in [1,4])",
    ship_rule="tier ships iff placement 95% CI lower bound > 2.500 AND per_move_ms <= 6000",
    calibration_N0=2.5,
    tiers=rows,
    winner_tier=best,
    verdict=("SEARCH BEATS aug_s0 at tier " + best) if best else
            "NULL: no search tier CI-separated above aug_s0 (2.500)",
)
json.dump(out, open(OUT, "w"), indent=2)
print(json.dumps(out, indent=2))
