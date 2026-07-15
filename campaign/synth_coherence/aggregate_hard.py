#!/usr/bin/env python3
"""Aggregate HARD-regime per-cell JSONs -> results/SYNTH_COHERENCE_HARD.json.
Same schema as the easy aggregate: gap_matrix, per-axis slopes, break-even
thresholds + a finer C=1 threshold slice (all eps present at C=1)."""
import os, json, glob

CELLDIR = "/root/synth_coherence/cells_hard"
OUT = "/root/synth_coherence/results/SYNTH_COHERENCE_HARD.json"
GRID_CS = [1, 2, 4, 8]
GRID_EPS = [0.0, 0.1, 0.2, 0.3, 0.4]  # regular grid axes for matrix/verdicts


def load():
    cells = {}
    for fp in glob.glob(os.path.join(CELLDIR, "cell_C*_eps*.json")):
        d = json.load(open(fp))
        cells[(d["C"], round(d["eps"], 2))] = d
    return cells


def slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def breakeven(xs, ys):
    for i in range(len(xs) - 1):
        if ys[i] <= 0 < ys[i + 1]:
            return xs[i] + (0 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    return None


def main():
    cells = load()
    if not cells:
        print("NO CELLS FOUND"); return
    any_cell = next(iter(cells.values()))
    config = {k: any_cell[k] for k in
              ["N_teachers", "M_students", "n_train", "n_test", "d", "K", "H_sub",
               "freq", "hidden", "n_layers", "epochs", "batch", "lr", "kd_T",
               "kd_alpha", "gt_world_seed"]}

    def gv(c, e, arm, field="mean"):
        d = cells.get((c, round(e, 2)))
        return None if d is None else d[arm][field]

    grid = {}
    for c in GRID_CS:
        for e in GRID_EPS:
            d = cells.get((c, round(e, 2)))
            if d is None:
                continue
            grid[f"C{c}_eps{e:.2f}"] = dict(
                C=c, eps=e,
                teacher_ens=d["teacher_ens"], student_ens=d["student_ens"],
                teacher_single=d["teacher_single"], student_single=d["student_single"],
                gap=d["gap"], seeds=d["seeds"])

    gap_matrix = {f"C{c}": {f"eps{e:.2f}": gv(c, e, "gap") for e in GRID_EPS} for c in GRID_CS}

    # verdict (a): gap rises with eps, per C
    a_all_C = {}
    for c in GRID_CS:
        xs = [e for e in GRID_EPS if cells.get((c, round(e, 2)))]
        ys = [gv(c, e, "gap") for e in xs]
        a_all_C[f"C{c}"] = dict(slope=slope(xs, ys),
                                monotone_rising=all(ys[i+1] >= ys[i] for i in range(len(ys)-1)),
                                breakeven_eps=breakeven(xs, ys),
                                gaps=list(zip(xs, ys)))
    # verdict (b): gap collapses with C, per eps
    b_all_eps = {}
    for e in GRID_EPS:
        xs = [c for c in GRID_CS if cells.get((c, round(e, 2)))]
        ys = [gv(c, e, "gap") for c in xs]
        b_all_eps[f"eps{e:.2f}"] = dict(slope=slope(xs, ys),
                                        monotone_collapsing=all(ys[i+1] <= ys[i] for i in range(len(ys)-1)),
                                        gaps=list(zip(xs, ys)))

    # finer threshold slice at C=1 across ALL eps present
    c1_eps = sorted({e for (c, e) in cells if c == 1})
    finer = [dict(eps=e, gap=gv(1, e, "gap"), gap_ci=gv(1, e, "gap", "ci95"),
                  teacher_ens=gv(1, e, "teacher_ens"), student_ens=gv(1, e, "student_ens"))
             for e in c1_eps]
    fxs = [p["eps"] for p in finer]; fys = [p["gap"] for p in finer]
    finer_slice = dict(C=1, points=finer, slope=slope(fxs, fys),
                       breakeven_eps=breakeven(fxs, fys))

    curve_a = a_all_C["C1"]
    e_hi = max(GRID_EPS)
    result = dict(
        regime="HARD (D=64,K=20,FREQ=5,H_sub=48)", config=config,
        Cs=GRID_CS, epsilons=GRID_EPS, grid=grid, gap_matrix=gap_matrix,
        verdict_a_all_C_rise=a_all_C,
        verdict_b_all_eps_collapse=b_all_eps,
        finer_threshold_slice_C1=finer_slice,
        verdicts=dict(
            a_gap_rises_with_eps_at_Cmin=dict(slope=curve_a["slope"], Cmin=1,
                holds=curve_a["slope"] > 0),
            b_gap_collapses_with_C_at_epsmax=dict(slope=b_all_eps[f"eps{e_hi:.2f}"]["slope"],
                eps=e_hi, holds=b_all_eps[f"eps{e_hi:.2f}"]["slope"] < 0)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print("WROTE", OUT, "  n_cells=%d" % len(cells))

    print("\nGAP matrix (student_ens - teacher_ens):")
    hdr = "      " + "".join(f"  eps={e:.2f}      " for e in GRID_EPS)
    print(hdr)
    for c in GRID_CS:
        row = f"C={c:<2}"
        for e in GRID_EPS:
            g = gv(c, e, "gap"); ci = gv(c, e, "gap", "ci95")
            row += f"  {g:+.4f}(+-{ci:.4f})" if g is not None else "     n/a       "
        print(row)
    print("\nVERDICT (a) gap-rises-with-eps slope per C:")
    for c in GRID_CS:
        v = a_all_C[f"C{c}"]
        print(f"  C={c}: slope={v['slope']:+.4f} mono_rising={v['monotone_rising']} breakeven_eps={v['breakeven_eps']}")
    print("VERDICT (b) gap-collapses-with-C slope per eps:")
    for e in GRID_EPS:
        v = b_all_eps[f"eps{e:.2f}"]
        print(f"  eps={e:.2f}: slope={v['slope']:+.5f} mono_collapsing={v['monotone_collapsing']}")
    print("\nFINER threshold slice at C=1 (breakeven_eps=%s):" % finer_slice["breakeven_eps"])
    for p in finer:
        print(f"  eps={p['eps']:.2f}  gap={p['gap']:+.4f}+/-{p['gap_ci']:.4f}  tE={p['teacher_ens']:.4f} sE={p['student_ens']:.4f}")


if __name__ == "__main__":
    main()
