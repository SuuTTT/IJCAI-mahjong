"""Seed-aware aggregator: pool ALL result cells (any seed), group by (eps, depth),
report mean gap +/- SD across seeds, plus per-seed values and per-arm winrates.
Reads the seed/eps/depth from inside each CELL.json (dir names not required)."""
import argparse, glob, json, os, math


def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def sd(xs):
    if len(xs) < 2: return 0.0
    m = mean(xs); return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
def slope(xs, ys):
    n = len(xs)
    if n < 2: return 0.0
    mx = mean(xs); my = mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = []
    for f in sorted(glob.glob(os.path.join(a.results, "*", "CELL.json"))):
        with open(f) as fh:
            cells.append(json.load(fh))
    # group by (eps, depth)
    groups = {}
    for c in cells:
        groups.setdefault((c["eps"], c["teacher_depth"]), []).append(c)

    curves = {}
    verdicts = {}
    epss = sorted({c["eps"] for c in cells})
    for eps in epss:
        rows = []
        depths = sorted({d for (e, d) in groups if e == eps})
        for d in depths:
            g = groups[(eps, d)]
            gaps = [c["gap_student_ens_minus_teacher_ens"] for c in g]
            te = [c["overall"]["teacher_ens"]["winrate"] for c in g]
            se = [c["overall"]["student_ens"]["winrate"] for c in g]
            seeds = [c["seed"] for c in g]
            rows.append({
                "teacher_depth": d, "n_seeds": len(g), "seeds": seeds,
                "gap_mean": round(mean(gaps), 4), "gap_sd": round(sd(gaps), 4),
                "gap_per_seed": [round(x, 4) for x in gaps],
                "teacher_ens_winrate_mean": round(mean(te), 4),
                "student_ens_winrate_mean": round(mean(se), 4),
            })
        curves[f"eps={eps}"] = rows
        ds = [r["teacher_depth"] for r in rows]
        gm = [r["gap_mean"] for r in rows]
        verdicts[f"eps={eps}"] = {
            "gap_vs_depth_slope": round(slope(ds, gm), 5),
            "monotone_rising": all(gm[i + 1] >= gm[i] for i in range(len(gm) - 1)),
            "gap_at_min_depth": gm[0] if gm else None,
            "gap_at_max_depth": gm[-1] if gm else None,
            "prediction_holds_positive_slope": slope(ds, gm) > 0,
        }
    out = {
        "domain": "Othello 6x6 (perfect information, alpha-beta minimax teacher)",
        "theory": "distill-then-ensemble beats teacher-ensembling more as the "
                  "imitated policy becomes stronger/more coherent; PREDICTION: "
                  "gap = student_ens - teacher_ens grows with teacher depth D.",
        "opponent_ladder": "minimax depth {1,3,5}, paired random openings",
        "n_cells_total": len(cells),
        "gap_vs_depth_curves_meanSD": curves,
        "verdicts": verdicts,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    for k, rows in curves.items():
        print(f"\n=== {k} ===")
        print(f"{'D':>3} {'teach_ens':>10} {'stud_ens':>9} {'gap_mean':>9} {'gap_sd':>7} {'seeds':>6}")
        for r in rows:
            print(f"{r['teacher_depth']:>3} {r['teacher_ens_winrate_mean']:>10.3f} "
                  f"{r['student_ens_winrate_mean']:>9.3f} {r['gap_mean']:>+9.4f} "
                  f"{r['gap_sd']:>7.4f} {r['n_seeds']:>6}")
        print(f"  slope={verdicts[k]['gap_vs_depth_slope']:+.5f} rising={verdicts[k]['monotone_rising']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
