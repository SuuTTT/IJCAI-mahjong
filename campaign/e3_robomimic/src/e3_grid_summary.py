#!/usr/bin/env python
"""Aggregate E3 robomimic eval jsons -> results/E3_ROBOMIMIC.json.

Per (task, demoset) cell:
  teacher_single = mean over 6 single-teacher success rates
  teacher_ens    = composition-average over 2 disjoint trios (trioA=s0s1s2, trioB=s3s4s5)
  student_single = mean over 3 single-student rates
  student_ens    = the s10/s11/s12 trio
  gap            = student_ens - teacher_ens
Prediction (E1/E2 mirror): gap larger on MH (noisy demos) than PH (clean).
"""
import argparse, glob, json, math, os

TASKS = ["lift", "can", "square"]
SETS = ["ph", "mh"]


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/E3_ROBOMIMIC.json")
    a = ap.parse_args()

    R = {}
    for fp in glob.glob(os.path.join(a.results, "*.json")):
        if os.path.basename(fp).startswith("E3_"):
            continue
        with open(fp) as f:
            j = json.load(f)
        if {"task", "demoset", "name", "success_rate"} <= set(j):
            R[(j["task"], j["demoset"], j["name"])] = j

    out = {"design": "E3 robomimic low_dim BC: 6 teachers -> 2 disjoint teacher-trios "
                     "vs 3 KD students (continuous KD: MSE to teacher-mean action, "
                     "alpha 0.7 soft / 0.3 hard) -> student-trio; "
                     ">=100 fixed-seed rollouts per point",
           "cells": {}, "prediction_check": {}}
    for task in TASKS:
        for ds in SETS:
            get = lambda n: R.get((task, ds, n))
            cell = {}
            ts = [get(f"teacher_s{i}") for i in range(6)]
            ss = [get(f"student_s{i}") for i in (10, 11, 12)]
            trios = [get("trioA"), get("trioB")]
            sens = get("student_ens")
            def rates(js):
                return [j["success_rate"] for j in js if j]
            cell["teacher_single_rates"] = rates(ts)
            cell["teacher_single_mean"] = (sum(rates(ts)) / len(rates(ts))) if rates(ts) else None
            cell["teacher_ens_rates"] = rates(trios)
            cell["teacher_ens_mean"] = (sum(rates(trios)) / len(rates(trios))) if rates(trios) else None
            cell["student_single_rates"] = rates(ss)
            cell["student_single_mean"] = (sum(rates(ss)) / len(rates(ss))) if rates(ss) else None
            cell["student_ens"] = sens["success_rate"] if sens else None
            if sens:
                cell["student_ens_ci95"] = wilson(sens["success_rate"], sens["n"])
            if cell["teacher_ens_mean"] is not None and cell["student_ens"] is not None:
                cell["gap_studentens_minus_teacherens"] = round(
                    cell["student_ens"] - cell["teacher_ens_mean"], 4)
            cell["n_missing"] = sum(x is None for x in ts + ss + trios + [sens])
            out["cells"][f"{task}_{ds}"] = cell

    for task in TASKS:
        gp = out["cells"][f"{task}_ph"].get("gap_studentens_minus_teacherens")
        gm = out["cells"][f"{task}_mh"].get("gap_studentens_minus_teacherens")
        out["prediction_check"][task] = {
            "gap_ph": gp, "gap_mh": gm,
            "mh_minus_ph": round(gm - gp, 4) if (gp is not None and gm is not None) else None,
            "prediction_holds": (gm > gp) if (gp is not None and gm is not None) else None,
        }
    diffs = [v["mh_minus_ph"] for v in out["prediction_check"].values()
             if v["mh_minus_ph"] is not None]
    out["prediction_check"]["mean_mh_minus_ph"] = (
        round(sum(diffs) / len(diffs), 4) if diffs else None)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["prediction_check"], indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
