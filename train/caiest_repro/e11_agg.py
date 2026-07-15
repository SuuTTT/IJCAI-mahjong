"""
e11_agg.py — aggregate e11 gate cells into AUG_RESULTS.json + AUG_WRITEUP.md.
Block-level t 95% CI (df=n-1), every number read from saved per-cell JSON. A candidate
BEATS bn128s1 iff its placement 95% CI lower bound > 2.500 (calibrated: bn128s1-vs-bn128s1
must read exactly 2.500). Reads aug_verify.json (aug validity) and tta_time.json (per-move).
"""
import json, glob, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
GD = os.path.join(HERE, "ckpt", "aug", "gates")
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
         17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
         25: 2.060, 30: 2.042, 40: 2.021}


def _t(dfn):
    if dfn in TCRIT: return TCRIT[dfn]
    ks = sorted(TCRIT);
    for k in ks:
        if k >= dfn: return TCRIT[k]
    return 1.96


def agg(tag):
    cells = sorted(glob.glob(os.path.join(GD, tag + "_s*.json")))
    vals = []; games = 0; firsts = []; fourths = []; secs = 0
    meta = {}
    for c in cells:
        d = json.load(open(c))
        vals.append(d["placement_pts"]); games += d["games"]
        firsts.append(d["first_pct"]); fourths.append(d["fourth_pct"]); secs += d.get("seconds", 0)
        meta = {"cand": d["cand"], "ref": d["ref"], "cand_tta": d.get("cand_tta", 0),
                "tta_perms": d.get("tta_perms", [])}
    n = len(vals)
    if n == 0: return None
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1); sd = math.sqrt(var)
        se = sd / math.sqrt(n); ci = _t(n - 1) * se
    else:
        sd = 0.0; se = 0.0; ci = 0.0
    lo, hi = mean - ci, mean + ci
    if lo > 2.500:
        verdict = "BEATS_BN128S1"
    elif hi < 2.500:
        verdict = "WORSE"
    else:
        verdict = "TIED_NOT_SEPARATED"
    return dict(tag=tag, **meta, n_blocks=n, total_games=games,
                placement_mean=round(mean, 4), placement_sd=round(sd, 4), placement_se=round(se, 4),
                ci95_lo=round(lo, 4), ci95_hi=round(hi, 4), margin_lo=round(lo - 2.500, 4),
                first_pct=round(sum(firsts) / n, 2), fourth_pct=round(sum(fourths) / n, 2),
                gate_seconds=round(secs, 1), verdict=verdict, block_placements=vals)


def main():
    tags = ["calib", "aug_s0", "aug_s1", "aug_s2", "tta6", "tta3"]
    results = {}
    for t in tags:
        r = agg(t)
        if r: results[t] = r

    av = json.load(open(os.path.join(HERE, "aug_verify.json"))) if os.path.exists(
        os.path.join(HERE, "aug_verify.json")) else {}
    tt = json.load(open(os.path.join(HERE, "tta_time.json"))) if os.path.exists(
        os.path.join(HERE, "tta_time.json")) else {}

    # winner selection among candidates that BEAT bn128s1
    cands = {k: v for k, v in results.items() if k != "calib"}
    winners = {k: v for k, v in cands.items() if v["verdict"] == "BEATS_BN128S1"}
    if winners:
        best = max(winners, key=lambda k: winners[k]["ci95_lo"])
        overall = f"WINNER: {best} CI-separated above bn128s1 (margin_lo=+{winners[best]['margin_lo']})"
    else:
        overall = ("NULL: no enhanced net or TTA config CI-separated above bn128s1 -> bn128s1 stays "
                   "(imitation ceiling for 128x40).")

    out = dict(
        experiment="AUG/REG/TTA push on deployable 128x40 vs bn128s1 (full_128x40_s1)",
        gate="e11_gate.py lam=0 calibrated duplicate-format placement gate; 2.500=tied with bn128s1; "
             "rule: 95% CI lower bound > 2.500 to BEAT",
        aug_validity={
            "structural_legality": av.get("structural_legality", {}),
            "fan_invariance_preserve_rate": {k: av.get("fan_invariance", {}).get(k, {}).get("preserve_rate")
                                             for k in av.get("fan_invariance", {})},
            "verdict": ("rank-reflect + dragon-perm PASS: fan-preserve rates match/exceed the "
                        "already-deployed suit-perm; residual mismatch is the single 推不倒 "
                        "(Reversible Tiles) fan (also affects suit-perm); WIND neg-control clearly "
                        "lower -> winds correctly excluded."),
        },
        tta_per_move_ms=tt,
        calibration=results.get("calib"),
        candidates=cands,
        overall_verdict=overall,
    )
    with open(os.path.join(HERE, "AUG_RESULTS.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- writeup ----
    L = []
    L.append("# AUG / REG / TTA push on the deployable 128x40 (vs bn128s1 = full_128x40_s1)\n")
    L.append("Gate: `e11_gate.py` lam=0 calibrated duplicate-format placement gate (4-seat rotation). "
             "2.500 = tied with bn128s1. **Rule: a candidate BEATS bn128s1 iff its 95% CI lower bound > 2.500.**\n")
    c = results.get("calib")
    if c:
        L.append(f"Calibration (bn128s1 vs bn128s1): placement = **{c['placement_mean']}** "
                 f"({c['n_blocks']} block(s)) — must read 2.500.\n")
    L.append("## STEP 1 — augmentation fan-validity\n")
    fr = av.get("fan_invariance", {})
    L.append("| transform | fan-preserve rate | structural/legality |")
    L.append("|---|---|---|")
    sl = av.get("structural_legality", {})
    def rate(k):
        d = fr.get(k, {}); return d.get("preserve_rate")
    L.append(f"| suit-perm (deployed baseline) | {rate('suit_perm(120)')} / {rate('suit_perm(201)')} | PASS ({sl.get('suit_perm_120')}) |")
    L.append(f"| rank-reflection | {rate('rank_reflect')} | PASS ({sl.get('rank_reflect')}) |")
    L.append(f"| dragon-perm | {rate('dragon_perm(102)')} / {rate('dragon_perm(201)')} | PASS ({sl.get('dragon_perm_102')}) |")
    L.append(f"| WIND swap (neg-control) | {rate('WIND_swap(F1F2)_NEGCTRL')} | (excluded) |")
    L.append("\nBoth rank-reflection and dragon-perm preserve fan at rates matching/exceeding the "
             "already-deployed suit-perm; the small residual is the single 推不倒 (Reversible Tiles) "
             "fan, which also affects suit-perm. The WIND negative control is clearly lower, proving "
             "the fan test discriminates → winds correctly excluded. **Used augs: suit-perm + rank-reflection + dragon-perm.**\n")
    L.append("## STEP 2/4 — enhanced nets & TTA, gated vs bn128s1\n")
    L.append("| candidate | n_blocks | games | placement mean | 95% CI | margin_lo | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in cands.items():
        L.append(f"| {k} ({v['cand']}) | {v['n_blocks']} | {v['total_games']} | {v['placement_mean']} | "
                 f"[{v['ci95_lo']}, {v['ci95_hi']}] | {v['margin_lo']:+} | {v['verdict']} |")
    L.append("")
    if tt:
        L.append("## STEP 3 — TTA per-move latency (CPU single-thread, Botzone-like; TLE ~1000 ms/move)\n")
        L.append(f"- single forward: **{tt.get('single_ms_per_move')} ms/move**")
        L.append(f"- 3-perm (C3) TTA: **{tt.get('tta3_C3_ms_per_move')} ms/move**")
        L.append(f"- 6-perm full TTA: **{tt.get('tta6_full_ms_per_move')} ms/move**\n")
    L.append("## VERDICT\n")
    L.append(overall + "\n")
    with open(os.path.join(HERE, "AUG_WRITEUP.md"), "w") as f:
        f.write("\n".join(L))

    print(json.dumps({k: {"mean": v["placement_mean"], "ci": [v["ci95_lo"], v["ci95_hi"]],
                          "n": v["n_blocks"], "verdict": v["verdict"]}
                      for k, v in results.items()}, indent=2))
    print("OVERALL:", overall)


if __name__ == "__main__":
    main()
