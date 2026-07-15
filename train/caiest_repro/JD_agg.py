"""
JD_agg.py — aggregate the JOINT-DEFENSE experiment.  Per lambda:
  * training: mean best-EMA val over 3 students, mean val chosen-danger (danger of argmax discard)
  * placement: 8 e12_ens_gate blocks (500 seeds = 2000 games each) of the 3-student ensemble vs
    aug_s0 -> mean over blocks +- 95% CI over block means (16000 games/lambda)
  * deal-in: 6 seed-disjoint 250-game DISim runs (paired game seeds ACROSS lambdas) -> pooled
    binomial rate +- 95% CI (n=1500/lambda)
Factual comparisons vs lambda=0 (plain-KD control): deal-in delta, placement delta, CI overlaps.
-> results/JOINT_DEFENSE.json
"""
import json, glob, os
import numpy as np

LAMS = ["0", "0.1", "0.3", "1.0"]


def train_stats(lam):
    fs = sorted(glob.glob(f"ckpt/jd/jd_lam{lam}_s*.json"))
    if not fs:
        return None
    rows = [json.load(open(f)) for f in fs]
    return dict(n_students=len(rows),
                val=round(float(np.mean([r["best_ema_val"] for r in rows])), 4),
                val_each=[r["best_ema_val"] for r in rows],
                chosen_danger=round(float(np.mean([r["val_chosen_danger"] for r in rows])), 4),
                pen_train_ema=round(float(np.mean([r["pen_train_ema"] for r in rows])), 4))


def gate_stats(lam):
    fs = sorted(glob.glob(f"results/jd_gate/lam{lam}_b*.json"))
    if not fs:
        return None
    pts = [json.load(open(f))["placement_pts"] for f in fs]
    games = sum(json.load(open(f))["games"] for f in fs)
    m = float(np.mean(pts)); sd = float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0
    ci = 1.96 * sd / max(len(pts) ** 0.5, 1)
    return dict(blocks=len(pts), games=games, placement=round(m, 4), ci=round(ci, 4),
                lo=round(m - ci, 4), hi=round(m + ci, 4), block_pts=[round(p, 4) for p in pts])


def dealin_stats(lam):
    fs = sorted(glob.glob(f"results/jd_dealin/lam{lam}_e*.json"))
    if not fs:
        return None
    D = sum(json.load(open(f))["dealins"] for f in fs)
    N = sum(json.load(open(f))["ngames"] for f in fs)
    W = sum(json.load(open(f))["wins"] for f in fs)
    p = D / N; ci = 1.96 * (p * (1 - p) / N) ** 0.5
    return dict(n=N, deal_in_rate=round(p, 4), ci95=round(ci, 4), lo=round(p - ci, 4),
                hi=round(p + ci, 4), win_rate=round(W / N, 4))


per = {}
for lam in LAMS:
    per[lam] = dict(train=train_stats(lam), gate=gate_stats(lam), dealin=dealin_stats(lam))

ctrl = per["0"]
lines = []
comp = {}
for lam in LAMS:
    t, g, di = per[lam]["train"], per[lam]["gate"], per[lam]["dealin"]
    if not (t and g and di):
        lines.append(f"lam={lam}: INCOMPLETE")
        continue
    line = (f"lam={lam}: val={t['val']} chosen_danger={t['chosen_danger']} | "
            f"placement={g['placement']}±{g['ci']} (16k games) | "
            f"deal_in={di['deal_in_rate']}±{di['ci95']} (n={di['n']})")
    if lam != "0" and ctrl["gate"] and ctrl["dealin"]:
        dd = round(di["deal_in_rate"] - ctrl["dealin"]["deal_in_rate"], 4)
        dp = round(g["placement"] - ctrl["gate"]["placement"], 4)
        di_ci_sep = di["hi"] < ctrl["dealin"]["lo"]              # deal-in strictly below control CI
        pl_ci_sep_down = g["hi"] < ctrl["gate"]["lo"]            # placement strictly below control
        comp[lam] = dict(dealin_minus_ctrl=dd, dealin_CI_below_ctrl=bool(di_ci_sep),
                         placement_minus_ctrl=dp, placement_CI_below_ctrl=bool(pl_ci_sep_down),
                         placement_ge_ctrl_point=bool(g["placement"] >= ctrl["gate"]["placement"]))
        line += f" | vs lam0: d_dealin={dd} (CI-sep={di_ci_sep}) d_placement={dp}"
    lines.append(line)

out = dict(
    experiment=("danger-penalized joint-defense KD: loss = 0.7*KD(T6) + 0.3*smoothedCE + "
                "lam*E[p_student(discard)*danger(s,discard)]; 128x40 students, 60k steps, "
                "3 seeds/lambda; danger head frozen danger4.bn.pkl scored on augmented obs"),
    control="lam=0 (plain KD, identical trainer/rng)",
    eval=dict(placement="e12_ens_gate 3-student ensemble vs aug_128x40_s0, 8 blocks x 500 seeds "
                        "(seed0=300000+b*500), CI over block means",
              dealin="DISim seat0 ensemble vs 3x aug_s0, 6x250 games, seeds 900000+e*100000+g "
                     "(disjoint across e, SHARED across lambdas = paired)"),
    per_lambda=per, comparisons_vs_lam0=comp, factual_lines=lines)
json.dump(out, open("results/JOINT_DEFENSE.json", "w"), indent=2)
print("\n".join(lines))
print("\nWROTE results/JOINT_DEFENSE.json")
