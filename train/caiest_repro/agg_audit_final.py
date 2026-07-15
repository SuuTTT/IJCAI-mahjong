#!/usr/bin/env python3
"""Aggregate audit_final/<cell>_b*.json -> results/AUDIT_FINAL_CELLS.json.
Fresh-wall cells for the papers' integrity/audit table: a03ens (alpha=0.3
students x3), kd1tens / kd2tens (teacher-count curve, 1- and 2-teacher KD
students x3). Same convention as ALLCELL_FRESH_REGATE.json: per cell the
across-block mean of placement_pts and the 95% CI lower bound
(mean - 1.96 * sd(ddof=1) / sqrt(n)). Calibration line = 2.500."""
import glob, json, math, os, re, time

CELLS = {
    "a03ens": {"desc": "3x alpha=0.3 KD students (paperx a03_s0/s1/s2), deploy ensemble vs aug_s0", "seed0_base": 1000000},
    "kd1tens": {"desc": "3x 1-teacher KD students (kdcurve kd1t_s0/s1/s3), deploy ensemble vs aug_s0", "seed0_base": 1030000},
    "kd2tens": {"desc": "3x 2-teacher KD students (kdcurve kd2t_s0/s1/s2), deploy ensemble vs aug_s0", "seed0_base": 1060000},
}

out = {"design": "12 disjoint fresh blocks x 2000 seeds (8000 duplicate games) "
                 "per cell, e12_ens_gate deploy mean-softmax rule vs "
                 "ckpt/aug/aug_128x40_s0.pkl; ci_lo = mean - 1.96*sd/sqrt(n); "
                 "calibration parity = 2.500",
       "date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
for cell, meta in CELLS.items():
    fs = sorted(glob.glob(f"audit_final/{cell}_b*.json"),
                key=lambda p: int(re.search(r"_b(\d+)\.json$", p).group(1)))
    vals, games = [], 0
    for f in fs:
        with open(f) as fh:
            d = json.load(fh)
        vals.append(d["placement_pts"])
        games += d.get("games", 0)
    if not vals:
        out[cell] = {"n": 0, "note": "no blocks yet", **meta}
        continue
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    out[cell] = {"n": n, "games_total": games,
                 "mean": round(mean, 4),
                 "ci_lo": round(mean - half, 4),
                 "ci_hi": round(mean + half, 4),
                 "sd_blocks": round(sd, 5),
                 "blocks": [round(v, 4) for v in vals],
                 "beats_parity_ci": mean - half > 2.500, **meta}

os.makedirs("results", exist_ok=True)
tmp = "results/AUDIT_FINAL_CELLS.json.tmp"
with open(tmp, "w") as f:
    json.dump(out, f, indent=2)
os.replace(tmp, "results/AUDIT_FINAL_CELLS.json")
print(json.dumps(out, indent=2))
