"""Build the distillation state buffer (E4).

Roll out each of the 6 teachers (stochastic sampling from its own policy)
for n_states/6 states each; then record EVERY teacher's softmax action
distribution on EVERY state in the mixed buffer.
"""
import argparse
import time

import numpy as np
import torch
from torch.distributions import Categorical

from common import load_net, VecMinAtar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--teacher-seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--n-states", type=int, default=100_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--rollout-seed-base", type=int, default=30000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(12345)
    np.random.seed(12345)

    teachers = [load_net(f"{args.ckpt_dir}/teacher_s{s}.pt", device)
                for s in args.teacher_seeds]
    per_teacher = args.n_states // len(teachers)

    all_states = []
    t0 = time.time()
    for ti, (s, net) in enumerate(zip(args.teacher_seeds, teachers)):
        venv = VecMinAtar(args.num_envs,
                          seed0=args.rollout_seed_base + s * 1000)
        collected = 0
        while collected < per_teacher:
            obs_np = venv.obs()
            take = min(venv.n, per_teacher - collected)
            all_states.append(obs_np[:take].astype(np.uint8))
            obs = torch.tensor(obs_np, device=device)
            with torch.no_grad():
                logits, _ = net(obs)
                actions = Categorical(logits=logits).sample().cpu().numpy()
            venv.step(actions)
            collected += take
        print(f"teacher s{s}: collected {collected} states "
              f"({time.time()-t0:.0f}s)", flush=True)

    states = np.concatenate(all_states, axis=0)  # (N,10,10,C) uint8
    n = states.shape[0]

    # every teacher's action distribution on every buffer state
    probs = np.zeros((len(teachers), n, teachers[0].pi.out_features),
                     np.float32)
    bs = 4096
    with torch.no_grad():
        for ti, net in enumerate(teachers):
            for i in range(0, n, bs):
                obs = torch.tensor(states[i:i + bs].astype(np.float32),
                                   device=device)
                logits, _ = net(obs)
                probs[ti, i:i + bs] = torch.softmax(logits, -1).cpu().numpy()

    mean_probs = probs.mean(axis=0)
    np.savez_compressed(
        args.out, states=states, teacher_probs=probs.astype(np.float16),
        mean_probs=mean_probs.astype(np.float32),
        teacher_seeds=np.array(args.teacher_seeds),
        rollout_seed_base=np.array(args.rollout_seed_base))
    print(f"DONE buffer: {n} states from {len(teachers)} teachers "
          f"({per_teacher}/teacher), saved to {args.out} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
