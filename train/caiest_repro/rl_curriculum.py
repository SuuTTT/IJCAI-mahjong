"""
rl_curriculum.py — Curriculum-learning RL for Chinese Standard Mahjong (reproduces the PKU thesis,
郑启帆/李文新 2024). The core idea: seed each RL episode from a near-win hand, starting easy
(tenpai, ~85% winnable -> dense reward) and increasing difficulty by shanten distance, so the agent
escapes the sparse-reward parity trap that kills vanilla self-play.

Stages (each warm-starts from the previous, Fig 3.6):
  tenpai(0) -> 0..1 shanten -> 0..2 -> 0..3 -> random full deal
Reward (Algorithm 1 + terminal): normalized game score for the seat, MINUS a penalty for each
Chi/Peng/Gang (discourages the 盲目吃碰 over-melding the thesis warns about). KL-to-SL leash keeps
the policy from drifting off the strong SL base. Promotion gated on the diverse gauntlet (#23).

  python3 rl_curriculum.py --base arch_ck/explore/resbn40_distill100b.pkl \
      --states /tmp/curriculum_states.pkl --out arch_ck/explore/resbn40_cl.pkl \
      --stage-iters 30 --actors 18 --games-per-actor 4 --gauntlet-games 8
"""
import os, sys, json, argparse, time, random, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import multiprocessing as mp

CURPATH = '/tmp/cl_cur.pkl'; POOLDIR = '/tmp/clpool'
# caiest action offsets (feature.py): meld = Chi[36] Peng[99] Gang[133]
MELD_LO, MELD_HI = 36, 167
STATES = None

def _load_states():
    global STATES
    if STATES is None:
        STATES = pickle.load(open(os.environ['CL_STATES'], 'rb'))
    return STATES

def actor_play(arg):
    seed, n_games, blocks, kmax, meld_pen = arg
    import torch as T
    T.set_num_threads(1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models_explore import ResBNCNN
    from sim_cnn import Sim
    rng = random.Random(seed)
    states = _load_states()
    # curriculum hand pool for this stage: all hands with shanten in [0, kmax] (kmax=-1 -> random deal)
    pool_hands = [] if kmax < 0 else [h for k in range(kmax + 1) for h in states.get(k, [])]
    cur = ResBNCNN(channels=128, blocks=blocks); cur.load_state_dict(T.load(CURPATH, map_location='cpu')); cur.eval()
    opp = ResBNCNN(channels=128, blocks=blocks); opp.eval()
    opp_files = sorted(glob.glob(os.path.join(POOLDIR, '*.pkl')))

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

    data = []
    for g in range(n_games):
        pf = opp_files[0] if (not opp_files or rng.random() < 0.4) else rng.choice(opp_files[-6:])
        if opp_files: opp.load_state_dict(T.load(pf, map_location='cpu'))
        store = []
        seat = 0
        kw = dict(seed=seed * 1000 + g, quan=0, learner_seats=[seat], cnn=True)
        if pool_hands:
            kw['seed_hand'] = rng.choice(pool_hands); kw['seed_seat'] = seat
        pols = [greedy(opp)] * 4; pols[seat] = sample_pol(store)
        sim = Sim(pols, **kw)
        sim.play()
        term = sim.scores[seat] / 8.0                       # +6 for an 8-fan self-draw; deal-in negative
        for row in store:
            r = term - (meld_pen if MELD_LO <= row[2] < MELD_HI else 0.0)   # Algorithm-1 anti-meld
            data.append(row + [r])
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
    ap.add_argument('--base', required=True); ap.add_argument('--states', required=True)
    ap.add_argument('--blocks', type=int, default=40); ap.add_argument('--out', required=True)
    ap.add_argument('--stage-iters', type=int, default=30); ap.add_argument('--actors', type=int, default=18)
    ap.add_argument('--games-per-actor', type=int, default=4); ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--clip', type=float, default=0.2); ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--beta-kl', type=float, default=0.3)
    ap.add_argument('--kl-decay', type=float, default=0.99); ap.add_argument('--meld-pen', type=float, default=0.5)
    ap.add_argument('--gauntlet-games', type=int, default=8); ap.add_argument('--snap-every', type=int, default=10)
    a = ap.parse_args()
    os.environ['CL_STATES'] = a.states
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(POOLDIR, exist_ok=True); [os.remove(f) for f in glob.glob(POOLDIR + '/*.pkl')]
    sl = torch.load(a.base, map_location='cpu')
    torch.save(sl, os.path.join(POOLDIR, '00_sl.pkl'), _use_new_zipfile_serialization=False)
    from models_explore import ResBNCNN
    model = ResBNPV(blocks=a.blocks).to(dev); model.net.load_state_dict(sl)
    sl_net = ResBNCNN(channels=128, blocks=a.blocks).to(dev); sl_net.load_state_dict(sl); sl_net.eval()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); beta = a.beta_kl; snap = 1
    from gauntlet_eval import gauntlet_net
    def gnet():
        cpu = ResBNCNN(channels=128, blocks=a.blocks); cpu.load_state_dict(model.net.state_dict()); cpu.eval()
        return gauntlet_net(cpu, n_games=a.gauntlet_games)
    best = gnet(); torch.save(model.net.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"[gauntlet] START net={best:+d} (distill100b floor; promote only if beaten)", flush=True)
    pool = mp.Pool(a.actors)
    STAGES = [0, 1, 2, 3, -1]; names = {0: 'tenpai', 1: '0-1sh', 2: '0-2sh', 3: '0-3sh', -1: 'random'}
    it = 0
    for kmax in STAGES:
        for si in range(a.stage_iters):
            t0 = time.time(); it += 1
            torch.save(model.net.state_dict(), CURPATH, _use_new_zipfile_serialization=False)
            args = [(it * 1000 + i, a.games_per_actor, a.blocks, kmax, a.meld_pen) for i in range(a.actors)]
            data = [r for c in pool.map(actor_play, args) for r in c]
            if not data: continue
            obs = torch.from_numpy(np.stack([d[0] for d in data])).to(dev)
            mask = torch.from_numpy(np.stack([d[1] for d in data])).to(dev)
            act = torch.tensor([d[2] for d in data], device=dev); oldlp = torch.tensor([d[3] for d in data], device=dev)
            ret = torch.tensor([d[4] for d in data], device=dev)
            with torch.no_grad():
                _, vp = model(obs, mask)
                slp = torch.softmax(sl_net({'is_training': False, 'obs': {'observation': obs, 'action_mask': mask}}), -1)
            adv = ret - vp; adv = (adv - adv.mean()) / (adv.std() + 1e-6)
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
            print(f"[{names[kmax]}] it{it} ret={float(ret.mean()):+.3f} kl={float(kl):.3f} beta={beta:.3f} n={len(data)} ({time.time()-t0:.0f}s)", flush=True)
            if it % a.snap_every == 0:
                torch.save(model.net.state_dict(), os.path.join(POOLDIR, f'{snap:02d}.pkl'), _use_new_zipfile_serialization=False); snap += 1
        g = gnet()
        if g > best:
            best = g; torch.save(model.net.state_dict(), a.out, _use_new_zipfile_serialization=False)
            print(f"  [gauntlet] after {names[kmax]}: net={g:+d} NEW BEST -> saved {a.out}", flush=True)
        else:
            print(f"  [gauntlet] after {names[kmax]}: net={g:+d} (best {best:+d}, not promoted)", flush=True)
    pool.close(); print("DONE", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
