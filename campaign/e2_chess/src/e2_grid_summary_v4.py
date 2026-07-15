#!/usr/bin/env python3
"""E2 grid summary v4 -> results/E2_GRID_v4.json.

Recomputes the band gaps using ALL available evidence:
  teacher trios : trioA, trioB (teachers s0-s5, 200 games/level)
                  trioE, trioF (independent teachers s6-s11, 400 games/level)
  student trios : student400 (s10-s12, distilled from s0-s5, 400 games/level)
                  student2   (s20-s22, distilled from s6-s11, 400 games/level)
teacher_ens = games-weighted mean score over available teacher trios;
gap_<stu>  = score(<stu>) - teacher_ens, per level, averaged over levels.
student2 is the REPLICATION: students distilled from a fully independent
teacher pool — its own band-gradient verdict is reported separately.
Standalone (the v3 script only reads fixed names). Robust to missing files.
"""
import argparse, json, os

BANDS = ["0800-1200", "1600-2000", "2400plus"]  # low -> high skill
TEACHER_TRIOS = ["trioA", "trioB", "trioE", "trioF"]
STUDENT_TRIOS = ["student400", "student2"]


def verdict(gaps):  # gaps in band order low->high
    if any(g is None for g in gaps):
        return None
    if gaps[0] > gaps[1] > gaps[2]:
        return "SUPPORTED: gap strictly decreases with source rating (monotone)"
    if gaps[0] > gaps[2]:
        return ("PARTIAL: gap(lowest band) > gap(highest band) "
                "but not monotone across all three")
    if gaps[0] < gaps[1] < gaps[2]:
        return ("REVERSED (monotone): gap strictly INCREASES with source rating")
    return "NOT SUPPORTED: gap does not shrink as source rating rises"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/E2_GRID_v4.json")
    a = ap.parse_args()

    grid, missing = {}, []
    for band in BANDS:
        row = {}
        for pol in TEACHER_TRIOS + STUDENT_TRIOS + ["single"]:
            f = os.path.join(a.results, f"e2_eval_{band}_{pol}.json")
            if os.path.exists(f):
                with open(f) as fh:
                    row[pol] = json.load(fh)
            else:
                missing.append(f)
        ttrios = [p for p in TEACHER_TRIOS if p in row]
        strios = [p for p in STUDENT_TRIOS if p in row]
        if not ttrios or not strios:
            continue
        levels = sorted(row[ttrios[0]]["levels"].keys(), key=int)
        per_level = {}
        gaps = {s: [] for s in strios}
        for lv in levels:
            tw = [(row[p]["levels"][lv]["score"], row[p]["n_games_per_level"])
                  for p in ttrios]
            t_ens = sum(s * n for s, n in tw) / sum(n for _, n in tw)
            d = {"teacher_trios": {p: row[p]["levels"][lv]["score"] for p in ttrios},
                 "teacher_ens_weighted": round(t_ens, 4)}
            if "single" in row:
                d["single_teacher"] = row["single"]["levels"][lv]["score"]
            for s in strios:
                sc = row[s]["levels"][lv]["score"]
                g = sc - t_ens
                gaps[s].append(g)
                d[s] = sc
                d[f"gap_{s}"] = round(g, 4)
            per_level[lv] = d
        band_row = {
            "n_games_per_level": {p: row[p]["n_games_per_level"]
                                  for p in ttrios + strios},
            "teacher_trios_used": ttrios,
            "student_trios_used": strios,
            "levels": per_level,
            "trio_spread": round(max(
                abs(row[p]["levels"][lv]["score"] - row[q]["levels"][lv]["score"])
                for lv in levels for p in ttrios for q in ttrios), 4),
        }
        for s in strios:
            band_row[f"mean_gap_{s}"] = round(sum(gaps[s]) / len(gaps[s]), 4)
        pooled = [g for s in strios for g in gaps[s]]
        band_row["mean_gap_pooled"] = round(sum(pooled) / len(pooled), 4)
        grid[band] = band_row

    out = {"design": "E2 grid v4: teacher_ens = games-weighted mean over all "
                     "available teacher trios (trioA/B: s0-s5; trioE/F: "
                     "independent s6-s11); student trios student400 (s10-s12, "
                     "from s0-s5) and student2 (s20-s22 REPLICATION, from "
                     "s6-s11); gap = student_ens - teacher_ens per Stockfish "
                     "node level, averaged over levels",
           "prediction": "student-ens minus teacher-ens gap grows as source "
                         "rating drops",
           "bands_low_to_high": BANDS, "bands": grid, "missing": missing}

    keys = [("pooled", "mean_gap_pooled")] + \
           [(s, f"mean_gap_{s}") for s in STUDENT_TRIOS]
    for label, k in keys:
        gs = [grid[b].get(k) if b in grid else None for b in BANDS]
        out[f"gap_by_band_{label}"] = dict(zip(BANDS, gs))
        if all(g is not None for g in gs):
            out[f"gap_low_minus_high_{label}"] = round(gs[0] - gs[2], 4)
            out[f"verdict_{label}"] = verdict(gs)
    v1 = out.get("verdict_student400")
    v2 = out.get("verdict_student2")
    if v1 and v2:
        out["replication"] = (
            "REPLICATED: independent-teacher student trio shows the same "
            "band-gradient direction" if v1.split(":")[0] == v2.split(":")[0]
            else f"DIVERGES: student400 -> {v1} ; student2 -> {v2}")

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, a.out)
    print(json.dumps({k: out[k] for k in out if k.startswith(("gap_by_band",
          "gap_low_minus_high", "verdict", "replication"))}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
