"""Aggregate per-cell CELL.json into OTHELLO_DOMAIN.json with the
gap-vs-teacher-depth curve (the verdict for the distill-then-ensemble theory)."""
import argparse, glob, json, os


def slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = []
    for f in sorted(glob.glob(os.path.join(a.results, "*", "CELL.json"))):
        with open(f) as fh:
            cells.append(json.load(fh))
    by_eps = {}
    for c in cells:
        by_eps.setdefault(c["eps"], []).append(c)
    curves = {}
    verdicts = {}
    for eps, cs in by_eps.items():
        cs = sorted(cs, key=lambda c: c["teacher_depth"])
        rows = [
            {"teacher_depth": c["teacher_depth"],
             "teacher_ens_winrate": c["overall"]["teacher_ens"]["winrate"],
             "student_ens_winrate": c["overall"]["student_ens"]["winrate"],
             "teacher_single_winrate": c["overall"]["teacher_single"]["winrate"],
             "student_single_winrate": c["overall"]["student_single"]["winrate"],
             "gap_student_ens_minus_teacher_ens":
                 c["gap_student_ens_minus_teacher_ens"],
             "gap_by_opponent": c["gap_by_opponent"],
             "teacher_ens_ci95": c["overall"]["teacher_ens"]["ci95"],
             "student_ens_ci95": c["overall"]["student_ens"]["ci95"],
             "n_games_per_arm": c["overall"]["teacher_ens"]["n_games"]}
            for c in cs]
        curves[f"eps={eps}"] = rows
        ds = [r["teacher_depth"] for r in rows]
        gaps = [r["gap_student_ens_minus_teacher_ens"] for r in rows]
        verdicts[f"eps={eps}"] = {
            "gap_vs_depth_slope": round(slope(ds, gaps), 5),
            "monotone_rising": all(gaps[i + 1] >= gaps[i] for i in range(len(gaps) - 1)),
            "gap_at_min_depth": gaps[0] if gaps else None,
            "gap_at_max_depth": gaps[-1] if gaps else None,
            "prediction_holds_positive_slope": slope(ds, gaps) > 0,
        }
    summary = {
        "domain": "Othello 6x6 (perfect information, minimax teacher)",
        "theory": "distill-then-ensemble beats teacher-ensembling more as the "
                  "imitated policy becomes stronger/more coherent; PREDICTION: "
                  "gap = student_ens - teacher_ens grows with teacher search depth D.",
        "teacher": "alpha-beta minimax, positional+mobility heuristic; depth D "
                   "controls coherence/strength",
        "opponent_ladder": "minimax depth {1,3,5}, paired random openings",
        "n_cells": len(cells),
        "gap_vs_depth_curves": curves,
        "verdicts": verdicts,
    }
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)
    for k, curve in curves.items():
        print(f"\n=== {k} ===")
        print(f"{'D':>3} {'teach_ens':>10} {'stud_ens':>9} {'GAP':>9}")
        for row in curve:
            print(f"{row['teacher_depth']:>3} "
                  f"{row['teacher_ens_winrate']:>10.3f} "
                  f"{row['student_ens_winrate']:>9.3f} "
                  f"{row['gap_student_ens_minus_teacher_ens']:>+9.4f}")
        print(f"  slope={verdicts[k]['gap_vs_depth_slope']:+.5f} "
              f"rising={verdicts[k]['monotone_rising']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
