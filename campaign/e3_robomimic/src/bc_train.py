#!/usr/bin/env python
"""E3 robomimic BC teacher / KD-student trainer (low_dim).

Mirrors the E1 (CIFAR-N) / E2 (chess) distill-then-ensemble design in the
robotics BC domain. Natural noise axis = demo quality (PH proficient-human vs
MH multi-human mixed-quality demos).

Teacher: BC MLP trunk (1024-1024 ReLU) + GMM head (5 modes, softplus stds,
NLL loss) — robomimic's standard low_dim "BC" baseline architecture
(Mandlekar et al. 2021). --head mse gives the plain deterministic variant.

KD student — adaptation of the discrete 0.7-soft/0.3-hard KD recipe to
CONTINUOUS control: there is no softmax over actions to match, so the
distillation target is the 6-teacher mean of the teachers' EXPECTED actions
(GMM: sum_m w_m mu_m; deterministic teachers = their output). Targets are
precomputed once over the dataset (teachers are fixed):

    loss = alpha * MSE(E[student(x)], mean_k E[teacher_k(x)])
         + (1-alpha) * NLL(student(x), a_demo),   alpha = 0.7

Eval-time action (see bc_eval.py) = argmax-component mean (deterministic
low-noise analogue of robomimic's low_noise_eval sampling), so fixed-seed
rollouts are exactly paired across policies/ensembles.
"""
import argparse, math, os, time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


class BCPolicy(nn.Module):
    def __init__(self, in_dim, act_dim, head="gmm", hidden=(1024, 1024), modes=5):
        super().__init__()
        self.head, self.modes, self.act_dim = head, modes, act_dim
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        if head == "gmm":
            self.mu = nn.Linear(d, modes * act_dim)
            self.log_sig = nn.Linear(d, modes * act_dim)
            self.logits = nn.Linear(d, modes)
        else:
            self.out = nn.Linear(d, act_dim)

    def _gmm(self, x):
        z = self.trunk(x)
        B = z.shape[0]
        mu = self.mu(z).view(B, self.modes, self.act_dim)
        sig = F.softplus(self.log_sig(z).view(B, self.modes, self.act_dim)) + 1e-4
        logits = self.logits(z)
        return mu, sig, logits

    def nll(self, x, a):
        mu, sig, logits = self._gmm(x)
        ad = a.unsqueeze(1)
        comp = -0.5 * (((ad - mu) / sig) ** 2 + 2 * torch.log(sig)
                       + math.log(2 * math.pi)).sum(-1)
        logp = torch.logsumexp(F.log_softmax(logits, -1) + comp, dim=1)
        return -logp.mean()

    def expected(self, x):
        """E[a|x]: GMM weighted-component mean; deterministic head: output."""
        if self.head == "gmm":
            mu, _, logits = self._gmm(x)
            w = F.softmax(logits, -1).unsqueeze(-1)
            return (w * mu).sum(1)
        return self.out(self.trunk(x))

    def action(self, x):
        """Eval action: argmax-component mean (deterministic low-noise eval)."""
        if self.head == "gmm":
            mu, _, logits = self._gmm(x)
            idx = logits.argmax(-1)
            return mu[torch.arange(x.shape[0]), idx]
        return self.out(self.trunk(x))


def load_dataset(path):
    X, A = [], []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for d in demos:
            g = f["data"][d]
            obs = np.concatenate([np.asarray(g["obs"][k]) for k in OBS_KEYS], axis=1)
            X.append(obs.astype(np.float32))
            A.append(np.asarray(g["actions"]).astype(np.float32))
    return np.concatenate(X), np.concatenate(A), len(demos)


def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = BCPolicy(ck["in_dim"], ck["act_dim"], head=ck.get("head", "gmm"),
                 hidden=tuple(ck["hidden"]), modes=ck.get("modes", 5))
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    return m, ck["mu_obs"], ck["sigma_obs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--head", default="gmm", choices=["gmm", "mse"])
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--teacher_ckpts", default="",
                    help="comma-separated teacher ckpts -> KD student mode")
    ap.add_argument("--alpha", type=float, default=0.7)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, A, ndemo = load_dataset(a.data)
    mu_obs = X.mean(0)
    sigma_obs = np.clip(X.std(0), 1e-6, None)
    Xt = torch.from_numpy((X - mu_obs) / sigma_obs).to(device)
    At = torch.from_numpy(A).to(device)
    N, in_dim = Xt.shape
    act_dim = At.shape[1]
    print(f"dataset {a.data}: {ndemo} demos, {N} transitions, obs {in_dim}, "
          f"act {act_dim}, head {a.head}", flush=True)

    # KD mode: precompute teacher-mean expected-action targets (teachers fixed)
    tpaths = [p for p in a.teacher_ckpts.split(",") if p]
    Tt = None
    if tpaths:
        Xraw = torch.from_numpy(X).to(device)
        acc = torch.zeros_like(At)
        with torch.no_grad():
            for p in tpaths:
                tm, tmu, tsig = load_policy(p, device)
                tin = (Xraw - torch.from_numpy(tmu).to(device)) / torch.from_numpy(tsig).to(device)
                for i in range(0, N, 8192):
                    acc[i:i + 8192] += tm.expected(tin[i:i + 8192])
                del tm
        Tt = acc / len(tpaths)
        del Xraw, acc
        print(f"KD mode: {len(tpaths)} teachers, alpha={a.alpha}", flush=True)

    model = BCPolicy(in_dim, act_dim, head=a.head).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, N, (a.bs,), device=device)
        x, tgt = Xt[idx], At[idx]
        hard = model.nll(x, tgt) if a.head == "gmm" else ((model.expected(x) - tgt) ** 2).mean()
        if Tt is not None:
            kd = ((model.expected(x) - Tt[idx]) ** 2).mean()
            loss = a.alpha * kd + (1.0 - a.alpha) * hard
        else:
            loss = hard
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20000 == 0 or step == a.steps:
            print(f"step {step}/{a.steps} loss {loss.item():.5f} hard {hard.item():.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({
        "model": model.state_dict(), "mu_obs": mu_obs, "sigma_obs": sigma_obs,
        "obs_keys": OBS_KEYS, "in_dim": in_dim, "act_dim": act_dim,
        "hidden": [1024, 1024], "head": a.head, "modes": 5,
        "meta": {"data": a.data, "seed": a.seed, "steps": a.steps, "bs": a.bs,
                 "lr": a.lr, "alpha": a.alpha if tpaths else None,
                 "teachers": tpaths, "final_loss": float(loss.item()),
                 "kd_note": "continuous-KD: MSE of student expected action to "
                            "precomputed 6-teacher mean expected action (alpha soft) "
                            "+ NLL to demo action (1-alpha hard)"},
    }, a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
