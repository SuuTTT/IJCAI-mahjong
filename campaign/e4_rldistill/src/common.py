"""Shared model / env / eval utilities for E4 RL-distillation probe (MinAtar Breakout)."""
import numpy as np
import torch
import torch.nn as nn
from minatar import Environment

ENV_NAME = "breakout"
STICKY = 0.1          # standard MinAtar sticky-action prob (train AND eval)
MAX_EP_LEN = 5000     # safety cap


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ACNet(nn.Module):
    """Small conv actor-critic for MinAtar (10x10xC input)."""

    def __init__(self, in_channels=4, num_actions=6):
        super().__init__()
        self.conv = layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, stride=1))
        self.fc = layer_init(nn.Linear(16 * 8 * 8, 128))
        self.pi = layer_init(nn.Linear(128, num_actions), std=0.01)
        self.v = layer_init(nn.Linear(128, 1), std=1.0)

    def features(self, x):
        # x: (B, 10, 10, C) float in {0,1}
        x = x.permute(0, 3, 1, 2)
        h = torch.relu(self.conv(x))
        return torch.relu(self.fc(h.flatten(1)))

    def forward(self, x):
        h = self.features(x)
        return self.pi(h), self.v(h).squeeze(-1)


class VecMinAtar:
    """N independent MinAtar envs stepped in a python loop (MinAtar is cheap)."""

    def __init__(self, n, seed0, sticky=STICKY, max_ep_len=MAX_EP_LEN):
        self.envs = [Environment(ENV_NAME, sticky_action_prob=sticky)
                     for _ in range(n)]
        for i, e in enumerate(self.envs):
            e.seed(seed0 + i)
            e.reset()
        self.n = n
        self.max_ep_len = max_ep_len
        self.ep_ret = np.zeros(n, np.float64)
        self.ep_len = np.zeros(n, np.int64)
        self.in_channels = self.envs[0].state_shape()[2]
        self.num_actions = self.envs[0].num_actions()

    def obs(self):
        return np.stack([e.state() for e in self.envs]).astype(np.float32)

    def step(self, actions):
        rews = np.zeros(self.n, np.float32)
        dones = np.zeros(self.n, bool)
        finished = []  # list of (episode_return, episode_len)
        for i, e in enumerate(self.envs):
            r, term = e.act(int(actions[i]))
            self.ep_ret[i] += r
            self.ep_len[i] += 1
            rews[i] = r
            if term or self.ep_len[i] >= self.max_ep_len:
                finished.append((float(self.ep_ret[i]), int(self.ep_len[i])))
                e.reset()
                self.ep_ret[i] = 0.0
                self.ep_len[i] = 0
                dones[i] = True
        return rews, dones, finished


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ACNet(ck["in_channels"], ck["num_actions"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


@torch.no_grad()
def policy_probs(nets, obs_t):
    """Mean-softmax over a list of nets. obs_t: (B,10,10,C) tensor -> (B,A) probs."""
    ps = []
    for net in nets:
        logits, _ = net(obs_t)
        ps.append(torch.softmax(logits, dim=-1))
    return torch.stack(ps).mean(0)


@torch.no_grad()
def eval_greedy(nets, n_episodes, seed_start, device,
                sticky=STICKY, max_ep_len=MAX_EP_LEN):
    """Greedy (argmax of mean-softmax) eval; one fresh env per episode,
    env seed = seed_start + episode_index. Returns list of episode returns."""
    returns = []
    for ep in range(n_episodes):
        env = Environment(ENV_NAME, sticky_action_prob=sticky)
        env.seed(seed_start + ep)
        env.reset()
        ret, steps, term = 0.0, 0, False
        while not term and steps < max_ep_len:
            obs = torch.tensor(env.state().astype(np.float32),
                               device=device).unsqueeze(0)
            probs = policy_probs(nets, obs)
            a = int(probs.argmax(dim=-1).item())
            r, term = env.act(a)
            ret += r
            steps += 1
        returns.append(float(ret))
    return returns


def mean_ci95(xs):
    xs = np.asarray(xs, np.float64)
    m = float(xs.mean())
    ci = float(1.96 * xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return m, ci
