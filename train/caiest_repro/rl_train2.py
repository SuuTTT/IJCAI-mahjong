"""
rl_train2.py — RL fine-tune with the research's parity fixes (deepresearch.md §3):
  (1) MODEL POOL opponents (not a single frozen base): SL + learner snapshots (cap 20, FIFO),
      sampled non-uniformly (favor SL + recent) -> strategic diversity.
  (2) KL-to-SL leash: loss += beta_KL * KL(pi_RL || pi_SL), beta decayed over training -> keeps
      the policy near the competent SL base while it searches for real improvements.
  (3) advantage normalization, PPO clip 0.2, entropy 0.01, epochs 3 (PKU hyperparams).
Learner seats {0,2}; pool opponents at {1,3}. Reward = learner duplicate score / 48.

  python3 rl_train2.py --base arch_ck/explore/resbn40.pkl --blocks 40 --iters 200 --games 48 \
      --out arch_ck/explore/resbn40_rl2.pkl
"""
import os, sys, json, argparse, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from models_explore import ResBNCNN
from sim_cnn import Sim

class ResBNPV(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__()
        self.net = ResBNCNN(channels=channels, blocks=blocks)
        self.vhead = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 9, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, obs, mask):
        f = self.net.body(self.net.stem(obs.float()))
        inf = torch.clamp(torch.log(mask.float()), -1e38, 1e38)
        return self.net.foot(f) + inf, self.vhead(f).squeeze(-1)

def cnn_policy(model, dev):
    """ResBNCNN (dict input) greedy policy."""
    def fn(obs, mask):
        with torch.no_grad():
            lg = model({'is_training': False, 'obs': {
                'observation': torch.from_numpy(np.ascontiguousarray(obs)).to(dev),
                'action_mask': torch.from_numpy(np.ascontiguousarray(mask)).to(dev)}})
        return [int(lg.cpu().numpy().flatten().argmax())]
    return fn

def sample_policy(model, dev, store):
    def fn(obs, mask):
        o = torch.from_numpy(np.ascontiguousarray(obs)).to(dev); m = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
        with torch.no_grad():
            lg, v = model(o, m); p = torch.softmax(lg, -1); a = int(torch.multinomial(p, 1).item())
            logp = float(torch.log(p[0, a] + 1e-9))
        store.append([obs[0].astype(np.int8), mask[0], a, logp, float(v.item())])
        return [a]
    return fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--blocks', type=int, default=40)
    ap.add_argument('--iters', type=int, default=200); ap.add_argument('--games', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-5); ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01); ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--beta-kl', type=float, default=0.5); ap.add_argument('--kl-decay', type=float, default=0.985)
    ap.add_argument('--pool-cap', type=int, default=20); ap.add_argument('--snap-every', type=int, default=5)
    ap.add_argument('--eval-every', type=int, default=10); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sl = torch.load(a.base, map_location='cpu')
    model = ResBNPV(blocks=a.blocks).to(dev); model.net.load_state_dict(sl)
    sl_net = ResBNCNN(channels=128, blocks=a.blocks).to(dev); sl_net.load_state_dict(sl); sl_net.eval()
    opp = ResBNCNN(channels=128, blocks=a.blocks).to(dev); opp.eval()   # reused; weights swapped per game
    pool = [sl]                                                          # state_dicts; SL always present
    beta = a.beta_kl
    for it in range(a.iters):
        t0 = time.time(); model.eval(); data = []
        for g in range(a.games):
            # non-uniform opponent: 35% SL, else a recent-ish pool member
            sd = pool[0] if (random.random() < 0.35 or len(pool) == 1) else random.choice(pool[-8:])
            opp.load_state_dict(sd)
            store = []; learner = sample_policy(model, dev, store); fz = cnn_policy(opp, dev)
            sim = Sim([learner, fz, learner, fz], seed=it * a.games + g, quan=0, learner_seats=[0, 2], cnn=True)
            sim.play()
            r = (sim.scores[0] + sim.scores[2]) / 48.0
            for row in store: data.append(row + [r])
        if not data: continue
        obs = torch.from_numpy(np.stack([d[0] for d in data])).to(dev)
        mask = torch.from_numpy(np.stack([d[1] for d in data])).to(dev)
        act = torch.tensor([d[2] for d in data], device=dev)
        oldlp = torch.tensor([d[3] for d in data], device=dev)
        val = torch.tensor([d[4] for d in data], device=dev); ret = torch.tensor([d[5] for d in data], device=dev)
        adv = ret - val; adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        with torch.no_grad():
            sl_logits = sl_net({'is_training': False, 'obs': {'observation': obs, 'action_mask': mask}})
            sl_p = torch.softmax(sl_logits, -1)
        model.train()
        for _ in range(a.epochs):
            lg, v = model(obs, mask); p = torch.softmax(lg, -1)
            lp = torch.log(p.gather(1, act[:, None]).squeeze(1) + 1e-9)
            ratio = torch.exp(lp - oldlp)
            pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
            vl = F.mse_loss(v, ret)
            ent = -(p * torch.log(p + 1e-9)).sum(1).mean()
            kl = (p * (torch.log(p + 1e-9) - torch.log(sl_p + 1e-9))).sum(1).mean()   # KL(pi_RL||pi_SL)
            loss = pl + 0.5 * vl - a.ent * ent + beta * kl
            opt = getattr(main, '_opt', None) or torch.optim.Adam(model.parameters(), lr=a.lr); main._opt = opt
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        beta *= a.kl_decay
        print(f"iter {it+1}/{a.iters} reward={float(ret.mean()):+.3f} kl={float(kl):.4f} beta={beta:.3f} ({time.time()-t0:.0f}s, pool={len(pool)})", flush=True)
        if (it + 1) % a.snap_every == 0:
            pool.append({k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()})
            if len(pool) > a.pool_cap: pool.pop(1)   # FIFO, keep SL at [0]
        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            torch.save(model.net.state_dict(), a.out, _use_new_zipfile_serialization=False)
            print(f"  saved -> {a.out}", flush=True)
    torch.save(model.net.state_dict(), a.out, _use_new_zipfile_serialization=False); print("DONE", flush=True)

if __name__ == '__main__':
    main()
