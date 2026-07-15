"""
recipe_agg.py — aggregate RECIPE sweep gate cells -> RECIPE_RESULTS.json + RECIPE_WRITEUP.md.
Every number read from saved per-cell JSON. Reference = aug_s0 (current best deployable 128x40).
Calibrated gate: aug_s0-vs-aug_s0 must read 2.500. A config BEATS aug_s0 iff its placement
95% CI lower bound (block-level t, df=n-1) > 2.500. val-acc parsed from the training log DONE lines.
"""
import json, glob, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
GD = os.path.join(HERE, "ckpt", "recipe", "gates")
QUEUE = os.path.join(HERE, "recipe_queue.txt")
TRAINLOG = "/root/recipe_train.log"
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
         17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
         25: 2.060, 30: 2.042, 40: 2.021}


def _t(dfn):
    if dfn in TCRIT: return TCRIT[dfn]
    for k in sorted(TCRIT):
        if k >= dfn: return TCRIT[k]
    return 1.96


def flags_of():
    m = {}
    if os.path.exists(QUEUE):
        for ln in open(QUEUE):
            ln = ln.strip()
            if not ln or ln.startswith("#") or "|" not in ln:
                continue
            tag, fl = ln.split("|", 1)
            m[tag.strip()] = fl.strip()
    return m


def val_of():
    """parse best_ema_val per recipe tag from the training log DONE lines."""
    v = {}
    if os.path.exists(TRAINLOG):
        for ln in open(TRAINLOG, errors="ignore"):
            mt = re.search(r"DONE best_ema_val=([0-9.]+) -> ckpt/recipe/([A-Za-z0-9_]+)\.pkl", ln)
            if mt:
                v[mt.group(2)] = float(mt.group(1))
    return v


def agg(tag):
    cells = sorted(glob.glob(os.path.join(GD, tag + "_s*.json")))
    vals = []; games = 0; firsts = []; fourths = []; secs = 0.0; meta = {}
    for c in cells:
        d = json.load(open(c))
        vals.append(d["placement_pts"]); games += d["games"]
        firsts.append(d["first_pct"]); fourths.append(d["fourth_pct"]); secs += d.get("seconds", 0)
        meta = {"cand": d["cand"], "ref": d["ref"]}
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1); sd = math.sqrt(var)
        se = sd / math.sqrt(n); ci = _t(n - 1) * se
    else:
        sd = se = ci = 0.0
    lo, hi = mean - ci, mean + ci
    if lo > 2.500:
        verdict = "BEATS_AUGS0"
    elif hi < 2.500:
        verdict = "WORSE"
    else:
        verdict = "TIED_NOT_SEPARATED"
    return dict(tag=tag, **meta, n_blocks=n, total_games=games,
                placement_mean=round(mean, 4), placement_sd=round(sd, 4), placement_se=round(se, 4),
                ci95_lo=round(lo, 4), ci95_hi=round(hi, 4), margin_lo=round(lo - 2.500, 4),
                first_pct=round(sum(firsts) / n, 2), fourth_pct=round(sum(fourths) / n, 2),
                gate_seconds=round(secs, 1), verdict=verdict, beats_augs0=(verdict == "BEATS_AUGS0"),
                block_placements=vals)


def main():
    flags = flags_of(); vals = val_of()
    # discover tags: any gate cell prefix that is not 'calib'
    tags = set()
    for c in glob.glob(os.path.join(GD, "*_s*.json")):
        b = os.path.basename(c)
        m = re.match(r"(.+)_s\d+\.json$", b)
        if m and m.group(1) != "calib":
            tags.add(m.group(1))
    calib = agg("calib")
    cands = {}
    for t in sorted(tags):
        r = agg(t)
        if r:
            r["recipe_flags"] = flags.get(t, "")
            r["val_acc"] = vals.get(t)
            cands[t] = r
    ranked = sorted(cands.values(), key=lambda r: r["placement_mean"], reverse=True)
    winners = [r for r in ranked if r["beats_augs0"]]
    if winners:
        best = max(winners, key=lambda r: r["ci95_lo"])
        overall = (f"WINNER: {best['tag']} CI-separated above aug_s0 (margin_lo=+{best['margin_lo']}, "
                   f"val_acc={best['val_acc']}) -> new deploy candidate (real-field check pending).")
    else:
        overall = ("NULL: no recipe variant CI-separated above aug_s0 -> aug_s0 confirmed the "
                   "recipe optimum for the deployable 128x40.")

    out = dict(
        experiment="RECIPE sweep on deployable 128x40 vs aug_s0 (current best); vary training recipe axes",
        reference="aug_s0 = ckpt/aug/aug_128x40_s0.pkl (128x40); calibrated e11_gate lam=0, 2.500=tied; "
                  "BEAT iff placement 95% CI lower bound > 2.500",
        base_recipe="e11_train.py --channels 128 --blocks 40 --steps 130000 --lr 3e-4 --wd 1.5e-4 "
                    "--lsm 0.05 --ema 0.999 --warmup 2000 --bs 1024 --p_suit 0.8 --p_ref 0.5 --p_drag 0.5",
        calibration_augs0_vs_augs0=calib,
        n_configs_gated=len(cands),
        configs=cands,
        ranking=[{"tag": r["tag"], "val_acc": r["val_acc"], "placement_mean": r["placement_mean"],
                  "ci95": [r["ci95_lo"], r["ci95_hi"]], "margin_lo": r["margin_lo"],
                  "beats_augs0": r["beats_augs0"]} for r in ranked],
        overall_verdict=overall,
    )
    with open(os.path.join(HERE, "RECIPE_RESULTS.json"), "w") as f:
        json.dump(out, f, indent=2)

    L = []
    L.append("# RECIPE optimization sweep on the deployable 128x40 (vs aug_s0)\n")
    L.append("Reference = **aug_s0** (`ckpt/aug/aug_128x40_s0.pkl`), the current best deployable net. "
             "Gate: `e11_gate.py` lam=0 calibrated duplicate-format placement gate (4-seat rotation). "
             "2.500 = tied with aug_s0. **A config BEATS aug_s0 iff its placement 95% CI lower bound > 2.500.**\n")
    if calib:
        L.append(f"Calibration (aug_s0 vs aug_s0): placement = **{calib['placement_mean']}** "
                 f"({calib['n_blocks']} block(s)) — must read 2.500.\n")
    L.append("Base recipe: `--steps 130000 --lr 3e-4 --wd 1.5e-4 --lsm 0.05 --ema 0.999 --warmup 2000 "
             "--bs 1024 --p_suit 0.8 --p_ref 0.5 --p_drag 0.5` (each config changes only the listed axis).\n")
    L.append("## Ranked table (gate vs aug_s0)\n")
    L.append("| rank | config | recipe change | val_acc | n_blk | placement mean | 95% CI | margin_lo | beats aug_s0 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, 1):
        L.append(f"| {i} | {r['tag']} | `{r['recipe_flags']}` | {r['val_acc']} | {r['n_blocks']} | "
                 f"{r['placement_mean']} | [{r['ci95_lo']}, {r['ci95_hi']}] | {r['margin_lo']:+} | "
                 f"{'YES' if r['beats_augs0'] else 'no'} |")
    L.append("")
    L.append("## Verdict\n")
    L.append(overall + "\n")
    with open(os.path.join(HERE, "RECIPE_WRITEUP.md"), "w") as f:
        f.write("\n".join(L))
    print("configs gated:", len(cands), "| calib:", calib["placement_mean"] if calib else None)
    print("OVERALL:", overall)


if __name__ == "__main__":
    main()
