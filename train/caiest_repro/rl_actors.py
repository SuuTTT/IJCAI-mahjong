"""
rl_actors.py — PARALLEL self-play rollout (actor-learner) for the pool+KL RL fine-tune.
The rollout (game generation) is the bottleneck and is embarrassingly parallel, so we run many
ACTOR processes across local cores (CPU inference, no GPU contention). Each actor plays games
sampling opponents from the shared MODEL POOL (state_dicts written to /tmp/rlpool by the learner),
and returns trajectory rows. The LEARNER (this process) does the PPO + KL-to-SL update on GPU.
This is the distributed design from deepresearch.md §2 (PKU actor-learner + model pool); it extends
to vast.ai by running the same actor function on remote boxes.

  python3 rl_actors.py --base arch_ck/explore/resbn40.pkl --blocks 40 --iters 200 \
      --actors 22 --games-per-actor 3 --out arch_ck/explore/resbn40_rl3.pkl
"""
import os, sys, json, argparse, time, random, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import multiprocessing as mp

POOLDIR = '/tmp/rlpool'; CURPATH = '/tmp/rl_cur.pkl'

# ---- actor (worker) side: CPU torch inference, plays games, returns rows ----
def actor_play(arg):
    seed, n_games, blocks = arg
    import torch as T
    T.set_num_threads(1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models_explore import ResBNCNN
    from sim_cnn import Sim
    rng = random.Random(seed)
    cur = ResBNCNN(channels=128, blocks=blocks); cur.load_state_dict(T.load(CURPATH, map_location='cpu')); cur.eval()
    pool_files = sorted(glob.glob(os.path.join(POOLDIR, '*.pkl')))
    opp = ResBNCNN(channels=128, blocks=blocks); opp.eval()
    data = []
    def sample_pol(store):
        def fn(obs, mask):
            with T.no_grad():
                lg = cur({'is_training': False, 'obs': {'observation': T.from_numpy(np.ascontiguousarray(obs)),
                                                        'action_mask': T.from_numpy(np.ascontiguousarray(mask))}})
                p = T.softmax(lg, -1); a = int(T.multinomial(p, 1).item()); logp = float(T.log(p[0, a] + 1e-9))
            store.append([obs[0].astype(np.int8), mask[0], a, logp]); return [a]
        return fn
    def greedy(m):
        def fn(obs, mask):
            with T.no_grad():
                lg = m({'is_training': False, 'obs': {'observation': T.from_numpy(np.ascontiguousarray(obs)),
                                                      'action_mask': T.from_numpy(np.ascontiguousarray(mask))}})
            return [int(lg.numpy().flatten().argmax())]
        return fn
    for g in range(n_games):
        # 35% SL base (pool[0]), else recent
        pf = pool_files[0] if (rng.random() < 0.35 or len(pool_files) == 1) else rng.choice(pool_files[-8:])
        opp.load_state_dict(T.load(pf, map_location='cpu'))
        store = []
        sim = Sim([sample_pol(store), greedy(opp), sample_pol(store), greedy(opp)],
                  seed=seed * 1000 + g, quan=0, learner_seats=[0, 2], cnn=True)
        sim.play()
        r = (sim.scores[0] + sim.scores[2]) / 48.0
        for row in store: data.append(row + [r])
    return data

class ResBNPV(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__(); from models_explore import ResBNCNN
        self.net = ResBNCNN(channels=channels, blocks=blocks)
        self.vhead = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 9, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, obs, mask):
        f = self.net.body(self.net.stem(obs.float()))
        return self.net.foot(f) + torch.clamp(torch.log(mask.float()), -1e38, 1e38), self.vhead(f).squeeze(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--blocks', type=int, default=40)
    ap.add_argument('--iters', type=int, default=200); ap.add_argument('--actors', type=int, default=22)
    ap.add_argument('--games-per-actor', type=int, default=3); ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--clip', type=float, default=0.2); ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--beta-kl', type=float, default=0.5)
    ap.add_argument('--kl-decay', type=float, default=0.985); ap.add_argument('--pool-cap', type=int, default=20)
    ap.add_argument('--snap-every', type=int, default=5); ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(POOLDIR, exist_ok=True); [os.remove(f) for f in glob.glob(POOLDIR + '/*.pkl')]
    sl = torch.load(a.base, map_location='cpu')
    torch.save(sl, os.path.join(POOLDIR, '00_sl.pkl'), _use_new_zipfile_serialization=False)  # pool[0]=SL
    from models_explore import ResBNCNN
    model = ResBNPV(blocks=a.blocks).to(dev); model.net.load_state_dict(sl)
    sl_net = ResBNCNN(channels=128, blocks=a.blocks).to(dev); sl_net.load_state_dict(sl); sl_net.eval()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); beta = a.beta_kl; snap_id = 1
    pool = mp.Pool(a.actors)
    for it in range(a.iters):
        t0 = time.time()
        torch.save(model.net.state_dict(), CURPATH, _use_new_zipfile_serialization=False)  # current learner for actors
        args = [(it * 1000 + i, a.games_per_actor, a.blocks) for i in range(a.actors)]
        chunks = pool.map(actor_play, args)
        data = [row for c in chunks for row in c]
        if not data: continue
        obs = torch.from_numpy(np.stack([d[0] for d in data])).to(dev)
        mask = torch.from_numpy(np.stack([d[1] for d in data])).to(dev)
        act = torch.tensor([d[2] for d in data], device=dev); oldlp = torch.tensor([d[3] for d in data], device=dev)
        ret = torch.tensor([d[4] for d in data], device=dev)
        with torch.no_grad():
            _, vpred = model(obs, mask)
            slp = torch.softmax(sl_net({'is_training': False, 'obs': {'observation': obs, 'action_mask': mask}}), -1)
        adv = ret - vpred; adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        model.train()
        for _ in range(a.epochs):
            lg, v = model(obs, mask); p = torch.softmax(lg, -1)
            lp = torch.log(p.gather(1, act[:, None]).squeeze(1) + 1e-9); ratio = torch.exp(lp - oldlp)
            pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
            vl = F.mse_loss(v, ret); ent = -(p * torch.log(p + 1e-9)).sum(1).mean()
            kl = (p * (torch.log(p + 1e-9) - torch.log(slp + 1e-9))).sum(1).mean()
            loss = pl + 0.5 * vl - a.ent * ent + beta * kl
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); beta *= a.kl_decay
        print(f"iter {it+1}/{a.iters} reward={float(ret.mean()):+.3f} kl={float(kl):.4f} beta={beta:.3f} samples={len(data)} ({time.time()-t0:.0f}s)", flush=True)
        if (it + 1) % a.snap_every == 0:
            torch.save(model.net.state_dict(), os.path.join(POOLDIR, f'{snap_id:02d}.pkl'), _use_new_zipfile_serialization=False)
            snap_id += 1
            sn = sorted(glob.glob(POOLDIR + '/[0-9]*.pkl'))
            for f in sn[:-(a.pool_cap)] if len(sn) > a.pool_cap else []:
                if '00_sl' not in f: os.remove(f)
        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            torch.save(model.net.state_dict(), a.out, _use_new_zipfile_serialization=False); print(f"  saved -> {a.out}", flush=True)
    pool.close(); print("DONE", flush=True)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
