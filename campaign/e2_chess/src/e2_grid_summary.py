#!/usr/bin/env python3
"""Aggregate the 12 E2 eval JSONs into results/E2_GRID.json:
per-band table (single teacher / 2 disjoint teacher trios + their mean /
student ens / gap) and the gap-vs-band verdict for the prediction
"student-ens minus teacher-ens gap grows as source rating drops"."""
import argparse, json, os

BANDS = ["0800-1200", "1600-2000", "2400plus"]  # low -> high skill
POLICIES = ["single", "trioA", "trioB", "student"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/E2_GRID.json")
    a = ap.parse_args()

    grid, missing = {}, []
    for band in BANDS:
        row = {}
        for pol in POLICIES:
            f = os.path.join(a.results, f"e2_eval_{band}_{pol}.json")
            if not os.path.exists(f):
                missing.append(f)
                continue
            with open(f) as fh:
                row[pol] = json.load(fh)
        if len(row) < len(POLICIES):
            continue
        levels = sorted(row["single"]["levels"].keys(), key=int)
        per_level, gaps = {}, []
        for lv in levels:
            sc = {p: row[p]["levels"][lv]["score"] for p in POLICIES}
            t_ens = 0.5 * (sc["trioA"] + sc["trioB"])
            gap = sc["student"] - t_ens
            gaps.append(gap)
            per_level[lv] = {
                "single_teacher": sc["single"],
                "trioA": sc["trioA"], "trioB": sc["trioB"],
                "teacher_ens": round(t_ens, 4),
                "student_ens": sc["student"],
                "gap_student_minus_teacher_ens": round(gap, 4),
                "elo_single": row["single"]["levels"][lv]["elo_diff_vs_level"],
                "elo_trioA": row["trioA"]["levels"][lv]["elo_diff_vs_level"],
                "elo_trioB": row["trioB"]["levels"][lv]["elo_diff_vs_level"],
                "elo_student": row["student"]["levels"][lv]["elo_diff_vs_level"],
            }
        grid[band] = {
            "n_games_per_level": row["single"]["n_games_per_level"],
            "levels": per_level,
            "mean_gap": round(sum(gaps) / len(gaps), 4),
            "trio_spread": round(max(abs(per_level[lv]["trioA"] - per_level[lv]["trioB"])
                                     for lv in levels), 4),
        }

    out = {"prediction": "student-ens minus teacher-ens gap grows as source rating drops",
           "bands_low_to_high": BANDS, "bands": grid, "missing": missing}
    if all(b in grid for b in BANDS):
        g = [grid[b]["mean_gap"] for b in BANDS]
        out["gap_by_band"] = dict(zip(BANDS, g))
        out["gap_low_minus_high"] = round(g[0] - g[2], 4)
        if g[0] > g[1] > g[2]:
            v = "SUPPORTED: gap strictly decreases with source rating (monotone)"
        elif g[0] > g[2]:
            v = "PARTIAL: gap(lowest band) > gap(highest band) but not monotone across all three"
        else:
            v = "NOT SUPPORTED: gap does not shrink as source rating rises"
        out["verdict"] = v

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, a.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
