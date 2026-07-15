#!/usr/bin/env python
"""E3 mixture-averaging follow-up aggregator -> results/E3_MIXTURE.json.

Compares, per (task, demoset) cell, the three ensembles (trioA, trioB,
student_ens) under the two composition rules:
  action_mean : results/<task>_<ds>_<name>.json           (original runs)
  mixture     : results/<task>_<ds>_<name>_mixture.json   (--combine mixture)
Same 100 fixed-seed rollouts (seed_base 5000), so per-rollout outcomes are
PAIRED; discordant counts (mix_only / am_only) are reported per ensemble.

HYPOTHESIS ON RECORD: mixture-averaging (uniform mixture of member GMMs,
argmax weight*density component; the distribution-averaging mechanism carrier)
restores the student-ens advantage over teacher-ens on MH cells, which
action-mean composition destroyed (mean of multimodal actions is off-manifold).
"""
import argparse, json, math, os

TASKS = ["lift", "can", "square"]
SETS = ["ph", "mh"]
ENS = ["trioA", "trioB", "student_ens"]


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/E3_MIXTURE.json")
    a = ap.parse_args()

    out = {"design": "E3 mixture-averaging follow-up: same ensembles + same 100 "
                     "fixed-seed rollouts, composition action_mean vs mixture "
                     "(uniform mixture of member GMMs; act = mean of the "
                     "argmax weight*mixture-density pooled component)",
           "hypothesis": "mixture-averaging restores the student-ens advantage "
                         "over teacher-ens on MH cells (distribution-averaging "
                         "is the mechanism carrier)",
           "cells": {}, "hypothesis_check": {}, "missing": []}

    for task in TASKS:
        for ds in SETS:
            cell = {"ensembles": {}}
            am, mx = {}, {}
            for name in ENS:
                am[name] = load(os.path.join(a.results, f"{task}_{ds}_{name}.json"))
                mx[name] = load(os.path.join(a.results, f"{task}_{ds}_{name}_mixture.json"))
                for tag, j in (("action_mean", am[name]), ("mixture", mx[name])):
                    if j is None:
                        out["missing"].append(f"{task}_{ds}_{name} [{tag}]")
                e = {}
                if am[name]:
                    e["action_mean_sr"] = am[name]["success_rate"]
                if mx[name]:
                    e["mixture_sr"] = mx[name]["success_rate"]
                    e["mixture_ci95"] = wilson(mx[name]["success_rate"], mx[name]["n"])
                if am[name] and mx[name]:
                    e["mixture_minus_action_mean"] = round(
                        mx[name]["success_rate"] - am[name]["success_rate"], 4)
                    pa, pm = am[name].get("per_rollout"), mx[name].get("per_rollout")
                    if pa and pm and len(pa) == len(pm):
                        e["paired"] = {
                            "both": sum(1 for u, v in zip(pa, pm) if u and v),
                            "mix_only": sum(1 for u, v in zip(pa, pm) if v and not u),
                            "am_only": sum(1 for u, v in zip(pa, pm) if u and not v),
                            "neither": sum(1 for u, v in zip(pa, pm) if not u and not v),
                        }
                cell["ensembles"][name] = e

            for tag, src in (("action_mean", am), ("mixture", mx)):
                trios = [src[n]["success_rate"] for n in ("trioA", "trioB") if src[n]]
                stu = src["student_ens"]["success_rate"] if src["student_ens"] else None
                if trios:
                    cell[f"teacher_ens_mean_{tag}"] = round(sum(trios) / len(trios), 4)
                if trios and stu is not None:
                    cell[f"gap_studentens_minus_teacherens_{tag}"] = round(
                        stu - sum(trios) / len(trios), 4)
            ga = cell.get("gap_studentens_minus_teacherens_action_mean")
            gm = cell.get("gap_studentens_minus_teacherens_mixture")
            if ga is not None and gm is not None:
                cell["gap_mixture_minus_gap_action_mean"] = round(gm - ga, 4)
            out["cells"][f"{task}_{ds}"] = cell

    # hypothesis: on MH cells, gap(mixture) > gap(action_mean) and gap(mixture) > 0
    mh = {}
    for task in TASKS:
        c = out["cells"][f"{task}_mh"]
        ga = c.get("gap_studentens_minus_teacherens_action_mean")
        gm = c.get("gap_studentens_minus_teacherens_mixture")
        mh[task] = {
            "gap_action_mean": ga, "gap_mixture": gm,
            "restored_vs_action_mean": (gm > ga) if (ga is not None and gm is not None) else None,
            "student_advantage_positive": (gm > 0) if gm is not None else None,
        }
    out["hypothesis_check"]["mh_cells"] = mh
    vals = [v for v in mh.values() if v["restored_vs_action_mean"] is not None]
    if vals:
        n_rest = sum(v["restored_vs_action_mean"] for v in vals)
        n_pos = sum(bool(v["student_advantage_positive"]) for v in vals)
        out["hypothesis_check"]["n_mh_cells_gap_increased"] = f"{n_rest}/{len(vals)}"
        out["hypothesis_check"]["n_mh_cells_gap_positive"] = f"{n_pos}/{len(vals)}"
        if n_rest == len(vals) and n_pos == len(vals):
            v = "SUPPORTED: mixture composition raises the student-ens gap above zero on all MH cells"
        elif n_rest == len(vals):
            v = "PARTIAL: mixture raises the gap on all MH cells but it is not positive everywhere"
        elif n_rest > 0:
            v = "MIXED: mixture raises the gap on some MH cells only"
        else:
            v = "NOT SUPPORTED: mixture composition does not raise the student-ens gap on MH cells"
        out["hypothesis_check"]["verdict"] = v
    # secondary: PH cells for contrast
    out["hypothesis_check"]["ph_cells"] = {
        task: {"gap_action_mean": out["cells"][f"{task}_ph"].get(
                   "gap_studentens_minus_teacherens_action_mean"),
               "gap_mixture": out["cells"][f"{task}_ph"].get(
                   "gap_studentens_minus_teacherens_mixture")}
        for task in TASKS}

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(out["hypothesis_check"], indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
