"""
rl_train.py — RL fine-tune the resbn CNN by self-play PPO, seeded from the SL base.
Learner plays seats {0,2}; FROZEN SL base plays {1,3} -> the learner is rewarded for BEATING
the strong base (avoids the co-evolution/passivity drift that wrecked MLP self-play). Reward =
the learner's duplicate game score. PPO clipped objective + value baseline + entropy.

  python3 rl_train.py --base arch_ck/explore/resbn40.pkl --blocks 40 --iters 200 --games 64 \
      --out arch_ck/explore/resbn40_rl.pkl
Benchmarks the RL policy vs the frozen SL base every --eval-every iters (the gate: RL must beat SL).
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from models_explore import ResBNCNN
from sim_cnn import Sim
from data.feature_agent import ACT

class ResBNPV(nn.Module):
    """resbn policy + a value head off the shared backbone."""
    def __init__(self, channels=128, blocks=40):
        super().__init__()
        self.net = ResBNCNN(channels=channels, blocks=blocks)
        self.vhead = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 9, 256), nn.ReLU(), nn.Linear(256, 1))
        self.ch = channels
    def forward(self, obs, mask):
        f = self.net.body(self.net.stem(obs.float()))
        logits = self.net.foot(f)
        inf = torch.clamp(torch.log(mask.float()), -1e38, 1e38)
        return logits + inf, self.vhead(f).squeeze(-1)

def greedy_policy(model, dev):
    """For the FROZEN ResBNCNN (dict input)."""
    def fn(obs, mask):
        with torch.no_grad():
            lg = model({'is_training': False, 'obs': {
                'observation': torch.from_numpy(np.ascontiguousarray(obs)).to(dev),
                'action_mask': torch.from_numpy(np.ascontiguousarray(mask)).to(dev)}})
        return [int(lg.cpu().numpy().flatten().argmax())]
    return fn

def sample_policy(model, dev, store):
    """Stochastic policy for rollout; records (obs,mask,act,logp,val) into `store`."""
    def fn(obs, mask):
        o = torch.from_numpy(np.ascontiguousarray(obs)).to(dev); m = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
        with torch.no_grad():
            lg, v = model(o, m)
            p = torch.softmax(lg, -1); a = int(torch.multinomial(p, 1).item())
            logp = float(torch.log(p[0, a] + 1e-9))
        store.append([obs[0].astype(np.int8), mask[0], a, logp, float(v.item())])
        return [a]
    return fn

def rollout(model, frozen, dev, games, seed0):
    """learner seats {0,2} (sampled) vs frozen {1,3}. Returns samples with terminal rewards."""
    data = []
    fz = greedy_policy(frozen, dev)
    for g in range(games):
        store = []
        learner = sample_policy(model, dev, store)
        pols = [learner, fz, learner, fz]
        sim = Sim(pols, seed=seed0 + g, quan=0, learner_seats=[0, 2], cnn=True)
        sim.play()
        r = (sim.scores[0] + sim.scores[2]) / 48.0   # learner's duplicate score, scaled
        for row in store:
            data.append(row + [r])
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--blocks', type=int, default=40)
    ap.add_argument('--iters', type=int, default=200); ap.add_argument('--games', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-5); ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01); ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--eval-every', type=int, default=10); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sl = torch.load(a.base, map_location='cpu')
    model = ResBNPV(blocks=a.blocks).to(dev); model.net.load_state_dict(sl)
    frozen = ResBNCNN(channels=128, blocks=a.blocks).to(dev); frozen.load_state_dict(sl); frozen.eval()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = -1e9
    for it in range(a.iters):
        t0 = time.time(); model.eval()
        data = rollout(model, frozen, dev, a.games, seed0=it * a.games)
        if not data: continue
        obs = torch.from_numpy(np.stack([d[0] for d in data])).to(dev)
        mask = torch.from_numpy(np.stack([d[1] for d in data])).to(dev)
        act = torch.tensor([d[2] for d in data], device=dev)
        oldlp = torch.tensor([d[3] for d in data], device=dev)
        val = torch.tensor([d[4] for d in data], device=dev)
        ret = torch.tensor([d[5] for d in data], device=dev)
        adv = (ret - val); adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        model.train()
        for _ in range(a.epochs):
            lg, v = model(obs, mask)
            p = torch.softmax(lg, -1); lp = torch.log(p.gather(1, act[:, None]).squeeze(1) + 1e-9)
            ratio = torch.exp(lp - oldlp)
            pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
            vl = F.mse_loss(v, ret)
            ent = -(p * torch.log(p + 1e-9)).sum(1).mean()
            loss = pl + 0.5 * vl - a.ent * ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        avgr = float(ret.mean())
        print(f"iter {it+1}/{a.iters} reward={avgr:+.3f} loss={float(loss):.3f} ({time.time()-t0:.0f}s, {len(data)} samples)", flush=True)
        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            torch.save(model.net.state_dict(), a.out, _use_new_zipfile_serialization=False)
            print(f"  saved RL policy -> {a.out}", flush=True)
    torch.save(model.net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE -> {a.out}", flush=True)

if __name__ == '__main__':
    main()
