"""CleanRL-style PPO teacher training on MinAtar Breakout (E4)."""
import argparse
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from common import ACNet, VecMinAtar


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training env seeds live in [seed*100, seed*100+num_envs) — disjoint from
    # eval ranges (>=10000) and buffer ranges (>=30000)
    venv = VecMinAtar(args.num_envs, seed0=args.seed * 100)
    net = ACNet(venv.in_channels, venv.num_actions).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    batch_size = args.num_envs * args.num_steps
    mb_size = batch_size // args.num_minibatches
    num_updates = args.total_steps // batch_size

    obs_buf = torch.zeros(args.num_steps, args.num_envs, 10, 10,
                          venv.in_channels, device=device)
    act_buf = torch.zeros(args.num_steps, args.num_envs, dtype=torch.long,
                          device=device)
    logp_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    rew_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    done_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    val_buf = torch.zeros(args.num_steps, args.num_envs, device=device)

    next_obs = torch.tensor(venv.obs(), device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    recent_returns = deque(maxlen=100)
    return_curve = []
    global_step = 0
    t0 = time.time()

    for update in range(1, num_updates + 1):
        for step in range(args.num_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                logits, value = net(next_obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp_buf[step] = dist.log_prob(action)
                val_buf[step] = value
            act_buf[step] = action
            rews, dones, finished = venv.step(action.cpu().numpy())
            for ret, _len in finished:
                recent_returns.append(ret)
            rew_buf[step] = torch.tensor(rews, device=device)
            next_obs = torch.tensor(venv.obs(), device=device)
            next_done = torch.tensor(dones.astype(np.float32), device=device)
            global_step += args.num_envs

        # GAE
        with torch.no_grad():
            _, next_value = net(next_obs)
            adv = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = (rew_buf[t] + args.gamma * nextvalues * nextnonterminal
                         - val_buf[t])
                lastgaelam = (delta + args.gamma * args.gae_lambda
                              * nextnonterminal * lastgaelam)
                adv[t] = lastgaelam
            ret_t = adv + val_buf

        b_obs = obs_buf.reshape(batch_size, 10, 10, venv.in_channels)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret_t.reshape(-1)
        b_val = val_buf.reshape(-1)

        idx = np.arange(batch_size)
        pl = vl = ent = 0.0
        for _ in range(args.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb_size):
                mb = idx[start:start + mb_size]
                logits, newval = net(b_obs[mb])
                dist = Categorical(logits=logits)
                newlogp = dist.log_prob(b_act[mb])
                entropy = dist.entropy().mean()
                ratio = (newlogp - b_logp[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef,
                                            1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_clip = b_val[mb] + torch.clamp(newval - b_val[mb],
                                                 -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max((newval - b_ret[mb]) ** 2,
                                         (v_clip - b_ret[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
                pl, vl, ent = pg_loss.item(), v_loss.item(), entropy.item()

        if update % 10 == 0 or update == num_updates:
            avg_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            sps = int(global_step / (time.time() - t0))
            return_curve.append({"step": global_step, "avg_return_100": avg_ret})
            print(f"seed={args.seed} update={update}/{num_updates} "
                  f"step={global_step} avg_return_100={avg_ret:.2f} "
                  f"pg_loss={pl:.4f} v_loss={vl:.4f} ent={ent:.3f} sps={sps}",
                  flush=True)
            torch.save({"state_dict": net.state_dict(),
                        "in_channels": venv.in_channels,
                        "num_actions": venv.num_actions,
                        "seed": args.seed, "global_step": global_step},
                       args.out + ".partial")

    torch.save({"state_dict": net.state_dict(),
                "in_channels": venv.in_channels,
                "num_actions": venv.num_actions,
                "seed": args.seed, "global_step": global_step,
                "return_curve": return_curve}, args.out)
    if os.path.exists(args.out + ".partial"):
        os.remove(args.out + ".partial")
    print(f"DONE teacher seed={args.seed} steps={global_step} "
          f"final_avg_return_100={np.mean(recent_returns):.2f} "
          f"wallclock={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
