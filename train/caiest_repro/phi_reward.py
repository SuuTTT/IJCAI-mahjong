"""
phi_reward.py — Suphx-style GLOBAL REWARD PREDICTOR Φ(state) -> expected final round score.
Two uses:
  1) gen   : self-play games, record (obs, final_score_for_that_seat) -> npz buffer.
  2) train : fit a small CNN Φ on that buffer (regression to normalized score).
The trained Φ gives a DENSE potential-based reward in rl_league (--phi): the per-step shaped
reward r̃_t = γ·Φ(s_{t+1}) − Φ(s_t), added to the sparse terminal score. This is the research's
fix (deepresearch §dense reward) for the high-variance terminal signal that left every RL
variant at parity. Potential-based shaping is policy-invariant in theory, so it can only help
credit assignment, not bias the optimum.

  python3 phi_reward.py gen   --model arch_ck/explore/resbn40.pkl --blocks 40 --games 2000 --actors 12 --out /tmp/phi_data.npz
  python3 phi_reward.py train --data /tmp/phi_data.npz --out arch_ck/explore/phi.pkl --epochs 6
"""
import os, sys, argparse, time, glob, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import multiprocessing as mp

MODELP = '/tmp/phi_genmodel.pkl'

class PhiNet(nn.Module):
    """Small ResNet-ish value-only net: (38,4,9) -> scalar score estimate."""
    def __init__(self, channels=64, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(38, channels, 3, 1, 1), nn.ReLU())
        body = []
        for _ in range(blocks):
            body += [nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU()]
        self.body = nn.Sequential(*body)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 9, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, obs):
        return self.head(self.body(self.stem(obs.float()))).squeeze(-1)


def _actor(arg):
    seed, n_games, blocks = arg
    import torch as T
    T.set_num_threads(1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models_explore import ResBNCNN
    from sim_cnn import Sim
    m = ResBNCNN(channels=128, blocks=blocks); m.load_state_dict(T.load(MODELP, map_location='cpu')); m.eval()
    def pol(seat_store, seat):
        def fn(obs, mask):
            with T.no_grad():
                lg = m({'is_training': False, 'obs': {'observation': T.from_numpy(np.ascontiguousarray(obs)),
                                                       'action_mask': T.from_numpy(np.ascontiguousarray(mask))}})
                p = T.softmax(lg, -1); a = int(T.multinomial(p, 1).item())
            seat_store.append(obs[0].astype(np.int8)); return [a]
        return fn
    rows = []
    for g in range(n_games):
        stores = [[], [], [], []]
        sim = Sim([pol(stores[i], i) for i in range(4)], seed=seed * 1000 + g, quan=0,
                  learner_seats=[0, 1, 2, 3], cnn=True)
        sim.play()
        for s in range(4):
            r = sim.scores[s] / 48.0
            for o in stores[s]:
                rows.append((o, r))
    return rows


def gen(a):
    sl = torch.load(a.model, map_location='cpu')
    torch.save(sl, MODELP, _use_new_zipfile_serialization=False)
    per = max(1, a.games // a.actors)
    pool = mp.Pool(a.actors)
    chunks = pool.map(_actor, [(i, per, a.blocks) for i in range(a.actors)])
    pool.close()
    rows = [r for c in chunks for r in c]
    obs = np.stack([r[0] for r in rows]).astype(np.int8)
    y = np.array([r[1] for r in rows], dtype=np.float32)
    np.savez_compressed(a.out, obs=obs, y=y)
    print(f"gen: {len(y)} states from ~{a.actors*per} games, y mean={y.mean():.3f} std={y.std():.3f} -> {a.out}", flush=True)


def train(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(a.data)
    obs = torch.from_numpy(d['obs']); y = torch.from_numpy(d['y'])
    n = len(y); idx = torch.randperm(n); ntr = int(n * 0.95)
    tr, va = idx[:ntr], idx[ntr:]
    net = PhiNet().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bs = 4096
    for ep in range(a.epochs):
        net.train(); perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            ob = obs[b].to(dev); yb = y[b].to(dev)
            pred = net(ob); loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(va), bs):
                vp.append(net(obs[va[i:i+bs]].to(dev)).cpu())
            vp = torch.cat(vp); vl = F.mse_loss(vp, y[va]).item()
            # baseline: predict mean
            bl = F.mse_loss(torch.full_like(y[va], float(y[tr].mean())), y[va]).item()
        print(f"ep {ep+1}/{a.epochs} val_mse={vl:.4f} (baseline {bl:.4f}, {'BETTER' if vl<bl else 'worse'})", flush=True)
    torch.save(net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"saved Φ -> {a.out}", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen'); g.add_argument('--model', required=True); g.add_argument('--blocks', type=int, default=40)
    g.add_argument('--games', type=int, default=2000); g.add_argument('--actors', type=int, default=12); g.add_argument('--out', required=True)
    t = sub.add_parser('train'); t.add_argument('--data', required=True); t.add_argument('--out', required=True); t.add_argument('--epochs', type=int, default=6)
    a = ap.parse_args()
    (gen if a.cmd == 'gen' else train)(a)
