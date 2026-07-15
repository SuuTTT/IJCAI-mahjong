"""E4 final eval: teacher-ensemble vs student-ensemble vs matched-K control.

All systems evaluated greedily (argmax of mean-softmax) on the SAME
200 eval env seeds (paired), disjoint from training / teacher-eval /
buffer-rollout seed ranges.
"""
import argparse
import json

import torch

from common import load_net, eval_greedy, mean_ci95


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--teacher-seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--student-seeds", type=int, nargs="+",
                   default=[10, 11, 12])
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--eval-seed-start", type=int, default=20000)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers = {s: load_net(f"{args.ckpt_dir}/teacher_s{s}.pt", device)
                for s in args.teacher_seeds}
    students = {s: load_net(f"{args.ckpt_dir}/student_s{s}.pt", device)
                for s in args.student_seeds}

    systems = {}
    for s in args.teacher_seeds:
        systems[f"teacher_s{s}_solo"] = [teachers[s]]
    systems["teacher_ensemble_K6"] = [teachers[s] for s in args.teacher_seeds]
    subset = args.teacher_seeds[:3]
    systems["teacher_subset_ensemble_K3_s" +
            "".join(map(str, subset))] = [teachers[s] for s in subset]
    for s in args.student_seeds:
        systems[f"student_s{s}_solo"] = [students[s]]
    systems["student_ensemble_M3"] = [students[s] for s in args.student_seeds]

    results = {}
    for name, nets in systems.items():
        rets = eval_greedy(nets, args.n_episodes, args.eval_seed_start, device)
        m, ci = mean_ci95(rets)
        results[name] = {"mean_return": m, "ci95": ci, "returns": rets}
        print(f"{name}: {m:.2f} +/- {ci:.2f}", flush=True)

    def g(name):
        return results[name]["mean_return"], results[name]["ci95"]

    te_m, te_ci = g("teacher_ensemble_K6")
    ctrl_name = [k for k in results if k.startswith("teacher_subset")][0]
    ct_m, ct_ci = g(ctrl_name)
    se_m, se_ci = g("student_ensemble_M3")
    t_solo = [results[f"teacher_s{s}_solo"]["mean_return"]
              for s in args.teacher_seeds]
    s_solo = [results[f"student_s{s}_solo"]["mean_return"]
              for s in args.student_seeds]

    def cmp(a_m, a_ci, b_m, b_ci):
        d = a_m - b_m
        sep = abs(d) > (a_ci + b_ci)
        return {"delta": d, "ci_separated": sep}

    verdict = {
        "hypothesis": ("Theory predicts distill-then-ensemble does NOT beat "
                       "teacher-ensembling when teachers are independently "
                       "seeded RL agents (incoherent multi-policy target); "
                       "an honest null is the expected outcome."),
        "student_ens_vs_teacher_ens_K6": cmp(se_m, se_ci, te_m, te_ci),
        "student_ens_vs_teacher_subset_K3": cmp(se_m, se_ci, ct_m, ct_ci),
        "teacher_ens_K6_vs_mean_teacher_solo": te_m - sum(t_solo) / len(t_solo),
        "summary": (
            f"student_ensemble_M3={se_m:.2f}+/-{se_ci:.2f} vs "
            f"teacher_ensemble_K6={te_m:.2f}+/-{te_ci:.2f} vs "
            f"matched-K control {ctrl_name}={ct_m:.2f}+/-{ct_ci:.2f}. "
            + ("Student ensemble BEATS the K=6 teacher ensemble beyond "
               "overlapping 95% CIs — contradicts the incoherence prediction."
               if (se_m - te_m) > (se_ci + te_ci) else
               "Student ensemble does NOT beat the K=6 teacher ensemble "
               "beyond overlapping 95% CIs — consistent with the theory "
               "prediction (incoherent multi-policy target).")
            + (" At matched ensemble size K=3, student ensemble "
               + ("BEATS" if (se_m - ct_m) > (se_ci + ct_ci) else
                  ("LOSES TO" if (ct_m - se_m) > (se_ci + ct_ci) else
                   "is statistically indistinguishable from"))
               + " the 3-teacher subset ensemble.")),
    }

    out = {
        "experiment": "E4_RLDISTILL",
        "env": "minatar/breakout",
        "design": ("K=6 PPO teachers (independent seeds) -> mixture state "
                   "buffer -> M=3 students distilled by KL to MEAN teacher "
                   "action distribution -> compare ensembles"),
        "teacher_seeds": args.teacher_seeds,
        "student_seeds": args.student_seeds,
        "n_episodes": args.n_episodes,
        "eval_seed_range": [args.eval_seed_start,
                            args.eval_seed_start + args.n_episodes - 1],
        "seed_hygiene": {
            "teacher_training_env_seeds": "seed*100 .. seed*100+31",
            "teacher_solo_100ep_eval_seeds": "10000..10099 (results/teacher_sN.json)",
            "buffer_rollout_env_seeds": "30000+seed*1000 ..",
            "final_eval_env_seeds": f"{args.eval_seed_start}..{args.eval_seed_start + args.n_episodes - 1}",
        },
        "eval_policy": "greedy argmax of mean-softmax, sticky_action_prob=0.1",
        "teacher_solo_means": dict(zip(map(str, args.teacher_seeds), t_solo)),
        "student_solo_means": dict(zip(map(str, args.student_seeds), s_solo)),
        "systems": results,
        "verdict": verdict,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("VERDICT: " + verdict["summary"], flush=True)


if __name__ == "__main__":
    main()
