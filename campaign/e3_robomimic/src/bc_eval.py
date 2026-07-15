#!/usr/bin/env python
"""E3 robomimic rollout eval: success rate of a policy or ensemble.

Fixed eval seeds: rollout i seeds numpy globally with seed_base+i before
env.reset(), which fixes the robosuite placement-sampler initial state — so all
policies/ensembles see the SAME initial-state sequence.

Ensemble composition (--combine):
  action_mean (default, original behaviour):
      act = clip(mean_k policy_k(obs), -1, 1)
      where policy_k(obs) is each member's argmax-component GMM mean.
      (E1/E2 mirror: composition = average over member outputs. Averaging
      MULTIMODAL actions can be off-manifold — the E3 boundary condition.)
  mixture (distribution averaging — GMM heads only):
      The ensemble is the uniform mixture of the K member GMMs: pooled
      component (k,m) has mixture weight pi_{k,m} = softmax(logits_k)_m / K.
      EXACT ACTION RULE: score every pooled component by
          score(k,m) = pi_{k,m} * p_mix(mu_{k,m})
      where p_mix(x) = sum_{k',m'} pi_{k',m'} N(x; mu_{k',m'}, diag sig_{k',m'}^2)
      is the full pooled-mixture density, evaluated at that component's mean;
      act = clip(mu of the argmax-score component, -1, 1).
      This is the mixture analogue of the single-model argmax-component-mean
      rule (the density factor makes modes that several members support beat
      isolated high-weight modes), and it never averages across modes, so the
      action always lies on a member-proposed mode.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from bc_train import BCPolicy, OBS_KEYS, load_policy  # noqa

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

MIXTURE_RULE = ("uniform mixture of member GMMs; pooled component (k,m) weight "
                "pi=softmax(logits_k)_m/K; act = mean of argmax_{k,m} "
                "pi_{k,m}*p_mix(mu_{k,m}) component, p_mix = full pooled-mixture "
                "density with diagonal-covariance components; clipped to [-1,1]")


def mixture_action(policies, xt):
    """Distribution-averaging ensemble action (see module docstring)."""
    mus, sigs, logws = [], [], []
    for m, mu_o, sig_o in policies:
        mu, sig, logits = m._gmm((xt - mu_o) / sig_o)
        mus.append(mu[0])                              # (M, D)
        sigs.append(sig[0])
        logws.append(F.log_softmax(logits[0], -1))     # (M,)
    mu = torch.cat(mus, 0)                             # (C, D), C = K*M
    sig = torch.cat(sigs, 0)
    logw = torch.cat(logws, 0) - math.log(len(policies))
    D = mu.shape[1]
    # log N(mu_c ; mu_c', diag sig_c'^2) for all component pairs (rows = eval pts)
    d = (mu.unsqueeze(1) - mu.unsqueeze(0)) / sig.unsqueeze(0)          # (C, C, D)
    logn = (-0.5 * (d ** 2).sum(-1)
            - torch.log(sig).sum(-1).unsqueeze(0)
            - 0.5 * D * math.log(2 * math.pi))                          # (C, C)
    logpmix = torch.logsumexp(logw.unsqueeze(0) + logn, dim=1)          # (C,)
    score = logw + logpmix                     # log(pi_c * p_mix(mu_c))
    return mu[int(score.argmax())].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="hdf5 (for env metadata)")
    ap.add_argument("--ckpts", required=True, help="comma-separated ckpts (1=single, 3=ens)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--demoset", required=True)
    ap.add_argument("--rollouts", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seed_base", type=int, default=5000)
    ap.add_argument("--combine", default="action_mean",
                    choices=["action_mean", "mixture"],
                    help="ensemble composition rule (see module docstring)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(2)
    device = "cpu"  # 1x1024 MLP: CPU forward beats GPU launch latency per step

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": OBS_KEYS, "rgb": []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=a.data)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)

    policies = []
    for p in a.ckpts.split(","):
        m, mu, sig = load_policy(p, device)
        policies.append((m, torch.from_numpy(mu), torch.from_numpy(sig)))
    if a.combine == "mixture" and not all(m.head == "gmm" for m, _, _ in policies):
        raise SystemExit("--combine mixture requires GMM heads on all members")

    succ = []
    t0 = time.time()
    for i in range(a.rollouts):
        np.random.seed(a.seed_base + i)
        obs = env.reset()
        s = 0
        for t in range(a.horizon):
            x = np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel()
                                for k in OBS_KEYS])
            xt = torch.from_numpy(x).unsqueeze(0)
            with torch.no_grad():
                if a.combine == "mixture":
                    act = np.clip(mixture_action(policies, xt), -1.0, 1.0)
                else:
                    acts = [m.action((xt - mu) / sig)[0].numpy()
                            for m, mu, sig in policies]
                    act = np.clip(np.mean(acts, axis=0), -1.0, 1.0)
            obs, r, done, _ = env.step(act)
            if env.is_success()["task"]:
                s = 1
                break
        succ.append(s)
        if (i + 1) % 20 == 0:
            print(f"rollout {i+1}/{a.rollouts} running_sr {np.mean(succ):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = {
        "task": a.task, "demoset": a.demoset, "name": a.name,
        "ckpts": a.ckpts.split(","), "k": len(policies),
        "combine": a.combine,
        "n": a.rollouts, "successes": int(np.sum(succ)),
        "success_rate": float(np.mean(succ)), "per_rollout": succ,
        "horizon": a.horizon, "seed_base": a.seed_base,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if a.combine == "mixture":
        out["combine_rule"] = MIXTURE_RULE
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(f"DONE {a.name}: success_rate {out['success_rate']:.3f} "
          f"({out['successes']}/{a.rollouts}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
