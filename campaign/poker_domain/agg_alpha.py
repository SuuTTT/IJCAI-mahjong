"""Aggregate the alpha-sensitivity probe: for each (alpha, eps), report the
student-ensemble advantage AND the single-net distillation denoising
(teacher_single - student_single), mean +/- 95% CI over seeds."""
import json, glob, math
from collections import defaultdict


def mean_ci(xs):
    n = len(xs); m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def main():
    groups = defaultdict(list)
    for f in sorted(glob.glob("results_alpha/a*_eps*_s*/CELL.json")):
        d = json.load(open(f))
        al = float(f.split("results_alpha/a")[1].split("_")[0])
        groups[(al, d["eps"])].append(d)

    rows = []
    print("alpha eps  n  ens_adv(+CI)          single_denoise(+CI)      t_ens  s_ens")
    for (al, eps) in sorted(groups):
        cs = groups[(al, eps)]
        adv = [c["advantage_student_ens_over_teacher_ens"] for c in cs]
        den = [c["exploitability"]["teacher_single"] - c["exploitability"]["student_single"] for c in cs]
        te = [c["exploitability"]["teacher_ens"] for c in cs]
        se = [c["exploitability"]["student_ens"] for c in cs]
        ma, ca = mean_ci(adv); md, cd = mean_ci(den)
        rows.append({"alpha": al, "eps": eps, "n_seeds": len(cs),
                     "ens_advantage_mean": round(ma, 5), "ens_advantage_ci95": round(ca, 5),
                     "single_denoise_mean": round(md, 5), "single_denoise_ci95": round(cd, 5),
                     "teacher_ens_mean": round(mean_ci(te)[0], 5),
                     "student_ens_mean": round(mean_ci(se)[0], 5),
                     "ens_advantage_per_seed": [round(x, 5) for x in adv],
                     "single_denoise_per_seed": [round(x, 5) for x in den]})
        print(f"{al:<5} {eps:<4} {len(cs):<2} {ma:+.5f} +/-{ca:.5f}   "
              f"{md:+.5f} +/-{cd:.5f}   {mean_ci(te)[0]:.3f}  {mean_ci(se)[0]:.3f}")
    json.dump({"probe": "alpha_sensitivity", "rows": rows},
              open("POKER_ALPHA_PROBE.json", "w"), indent=2)
    print("wrote POKER_ALPHA_PROBE.json")


if __name__ == "__main__":
    main()
