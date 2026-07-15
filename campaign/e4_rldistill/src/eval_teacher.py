"""Per-teacher greedy eval (E4). Eval seed range disjoint from training seeds."""
import argparse
import json

import torch

from common import load_net, eval_greedy, mean_ci95


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--eval-seed-start", type=int, default=10000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(args.ckpt, device)
    returns = eval_greedy([net], args.n_episodes, args.eval_seed_start, device)
    m, ci = mean_ci95(returns)
    result = {
        "experiment": "E4_teacher_solo_eval",
        "env": "minatar/breakout",
        "train_seed": args.seed,
        "n_episodes": args.n_episodes,
        "eval_seed_range": [args.eval_seed_start,
                            args.eval_seed_start + args.n_episodes - 1],
        "eval_policy": "greedy(argmax softmax), sticky_action_prob=0.1",
        "mean_return": m,
        "ci95": ci,
        "returns": returns,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"teacher s{args.seed}: mean_return={m:.2f} +/- {ci:.2f} "
          f"(n={args.n_episodes})", flush=True)


if __name__ == "__main__":
    main()
