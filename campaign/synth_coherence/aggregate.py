#!/usr/bin/env python3
"""Aggregate per-cell JSONs into SYNTH_COHERENCE.json with the two verdict curves."""
import os, json, glob, math

CELLDIR = "/root/synth_coherence/cells"
OUT = "/root/synth_coherence/SYNTH_COHERENCE.json"


def load():
    cells = {}
    for fp in glob.glob(os.path.join(CELLDIR, "cell_C*_eps*.json")):
        d = json.load(open(fp))
        cells[(d["C"], round(d["eps"], 2))] = d
    return cells


def main():
    cells = load()
    if not cells:
        print("NO CELLS FOUND")
        return
    Cs = sorted({c for (c, e) in cells})
    Es = sorted({e for (c, e) in cells})
    any_cell = next(iter(cells.values()))
    config = {k: any_cell[k] for k in
              ["N_teachers", "M_students", "n_train", "n_test", "d", "K", "H_sub",
               "hidden", "n_layers", "epochs", "batch", "lr", "kd_T", "kd_alpha",
               "gt_world_seed"]}

    def g(c, e, arm, field="mean"):
        d = cells.get((c, e))
        if d is None:
            return None
        return d[arm][field]

    grid = {}
    for c in Cs:
        for e in Es:
            d = cells.get((c, e))
            if d is None:
                continue
            grid[f"C{c}_eps{e:.2f}"] = dict(
                C=c, eps=e,
                teacher_ens=d["teacher_ens"], student_ens=d["student_ens"],
                teacher_single=d["teacher_single"], student_single=d["student_single"],
                gap=d["gap"], seeds=d["seeds"],
            )

    # Verdict (a): at fixed (low) coherence C=1, gap vs eps -> should RISE with eps.
    c1 = min(Cs)  # most coherent
    curve_a = [dict(eps=e,
                    gap=g(c1, e, "gap"), gap_ci=g(c1, e, "gap", "ci95"),
                    teacher_ens=g(c1, e, "teacher_ens"),
                    student_ens=g(c1, e, "student_ens"))
               for e in Es if cells.get((c1, e))]

    # Verdict (b): at each fixed eps, gap vs C -> should COLLAPSE (toward 0) as C grows.
    curve_b = {}
    for e in Es:
        curve_b[f"eps{e:.2f}"] = [dict(C=c,
                                       gap=g(c, e, "gap"), gap_ci=g(c, e, "gap", "ci95"),
                                       teacher_ens=g(c, e, "teacher_ens"),
                                       student_ens=g(c, e, "student_ens"))
                                  for c in Cs if cells.get((c, e))]

    # simple verdict checks
    def slope(xs, ys):
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n; my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den if den else 0.0

    a_eps = [p["eps"] for p in curve_a]
    a_gap = [p["gap"] for p in curve_a]
    verdict_a_slope = slope(a_eps, a_gap)

    # (b) measured at the highest eps: gap collapse across C
    e_hi = max(Es)
    b_hi = curve_b[f"eps{e_hi:.2f}"]
    b_C = [p["C"] for p in b_hi]
    b_gap = [p["gap"] for p in b_hi]
    verdict_b_slope_hi = slope(b_C, b_gap)

    # 2D gap matrix + per-C rise slope + per-eps collapse slope + break-even eps
    gap_matrix = {f"C{c}": {f"eps{e:.2f}": g(c, e, "gap") for e in Es} for c in Cs}
    a_all_C = {}
    for c in Cs:
        xs = [e for e in Es if cells.get((c, e))]
        ys = [g(c, e, "gap") for e in xs]
        mono = all(ys[i + 1] >= ys[i] for i in range(len(ys) - 1))
        # break-even eps: linear interp where gap crosses 0
        be = None
        for i in range(len(xs) - 1):
            if ys[i] <= 0 < ys[i + 1]:
                be = xs[i] + (0 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
                break
        a_all_C[f"C{c}"] = dict(slope=slope(xs, ys), monotone_rising=mono, breakeven_eps=be)
    b_all_eps = {}
    for e in Es:
        xs = [c for c in Cs if cells.get((c, e))]
        ys = [g(c, e, "gap") for c in xs]
        mono = all(ys[i + 1] <= ys[i] for i in range(len(ys) - 1))
        b_all_eps[f"eps{e:.2f}"] = dict(slope=slope(xs, ys), monotone_collapsing=mono)

    result = dict(
        config=config,
        Cs=Cs, epsilons=Es,
        grid=grid,
        gap_matrix=gap_matrix,
        verdict_a_curve_gap_vs_eps_at_Cmin=curve_a,
        verdict_a_all_C_rise=a_all_C,
        verdict_b_curves_gap_vs_C_by_eps=curve_b,
        verdict_b_all_eps_collapse=b_all_eps,
        verdicts=dict(
            a_gap_rises_with_eps_at_Cmin=dict(
                slope=verdict_a_slope, Cmin=c1,
                holds=verdict_a_slope > 0,
                note="gap = student_ens - teacher_ens; positive slope => grows with noise"),
            b_gap_collapses_with_C_at_epsmax=dict(
                slope=verdict_b_slope_hi, eps=e_hi,
                holds=verdict_b_slope_hi < 0,
                note="negative slope => gap shrinks as target becomes incoherent"),
        ),
    )
    json.dump(result, open(OUT, "w"), indent=2)
    print("WROTE", OUT)
    print(f"n_cells={len(cells)}/{len(Cs)*len(Es)}")
    print("VERDICT (a) gap-vs-eps slope @C%d = %+.5f (holds=%s)" %
          (c1, verdict_a_slope, verdict_a_slope > 0))
    print("VERDICT (b) gap-vs-C slope @eps%.2f = %+.6f (holds=%s)" %
          (e_hi, verdict_b_slope_hi, verdict_b_slope_hi < 0))
    print("\ncurve (a) gap vs eps at C=%d:" % c1)
    for p in curve_a:
        print(f"  eps={p['eps']:.2f}  gap={p['gap']:+.4f}+/-{p['gap_ci']:.4f}  "
              f"tE={p['teacher_ens']:.4f} sE={p['student_ens']:.4f}")
    print("\ncurve (b) gap vs C at eps=%.2f:" % e_hi)
    for p in b_hi:
        print(f"  C={p['C']}  gap={p['gap']:+.4f}+/-{p['gap_ci']:.4f}  "
              f"tE={p['teacher_ens']:.4f} sE={p['student_ens']:.4f}")


if __name__ == "__main__":
    main()
