"""
rl_online.py — ONLINE SELF-PLAY POLICY-GRADIENT RL, gate-matched (the untested SOTA lever).

Objective == the calibrated duplicate gate (e11_gate.py): ONE learner seat vs THREE identical
frozen opponents; reward = per-deal PLACEMENT (5 - avg_rank on the deal scores, gate formula;
self-play parity == 2.500). This is the true contest objective, optimised directly.

Algorithm: PPO (clip) with a value baseline, ENTROPY bonus, and a KL-leash to the frozen
aug_s0 SL anchor (trust region so the competent BC policy can't collapse). Init from aug_s0.
Opponents = a self-play pool: frozen aug_s0 (always, prob p_anchor) + past learner snapshots
(PFSP-ish uniform over recent). Learner seat rotates over {0,1,2,3} to match the gate rotation.

CPU actors (multiprocessing, 1 thread each) do the rollouts; one GPU process does the PPO update
(coexists with the recipe sweep / arch experiment: light GPU, CPU-bounded, capped actor count).

Snapshots: every --snap-every iters, write a fused resbn_fused .pkl to --snapdir (gateable with
e11_gate.py default cand-kind) + a .bn.pkl into the opponent pool. Honors /root/STOP_RL.
"""
import os, sys, json, argparse, time, random, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import multiprocessing as mp

CUR_MAIN = '/tmp/rlon_main.pkl'      # current learner policy weights (ResBNCNN state_dict)
POOLDIR = '/tmp/rlon_pool'           # opponent pool: 00_anchor.pkl + m_XXXX.pkl snapshots
STOP = '/root/STOP_RL'

def placement(scores, seat):
    cs = scores[seat]
    greater = sum(1 for j in range(4) if scores[j] > cs)
    equal = sum(1 for j in range(4) if scores[j] == cs)
    avg_rank = greater + (equal + 1) / 2.0
    return 5.0 - avg_rank

# ---------------- actor (CPU worker) ----------------
def actor_play(arg):
    seed, n_games, blocks, p_anchor = arg
    import torch as T
    T.set_num_threads(1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models_explore import ResBNCNN
    from sim_cnn import Sim
    rng = random.Random(seed)
    cur = ResBNCNN(channels=128, blocks=blocks)
    cur.load_state_dict(T.load(CUR_MAIN, map_location='cpu')); cur.eval()
    opp = ResBNCNN(channels=128, blocks=blocks); opp.eval()

    def sample_pol(store):
        def fn(obs, mask):
            with T.no_grad():
                lg = cur({'is_training': False, 'obs': {
                    'observation': T.from_numpy(np.ascontiguousarray(obs)),
                    'action_mask': T.from_numpy(np.ascontiguousarray(mask))}})
                p = T.softmax(lg, -1); a = int(T.multinomial(p, 1).item())
                logp = float(T.log(p[0, a] + 1e-9))
            store.append([obs[0].astype(np.int8), mask[0], a, logp]); return [a]
        return fn

    def greedy(m):
        def fn(obs, mask):
            with T.no_grad():
                lg = m({'is_training': False, 'obs': {
                    'observation': T.from_numpy(np.ascontiguousarray(obs)),
                    'action_mask': T.from_numpy(np.ascontiguousarray(mask))}})
            return [int(lg.numpy().flatten().argmax())]
        return fn

    anchor = os.path.join(POOLDIR, '00_anchor.pkl')
    snaps = sorted(glob.glob(POOLDIR + '/m_*.pkl'))
    out = []
    for g in range(n_games):
        # opponent = anchor (aug_s0) with prob p_anchor, else a recent learner snapshot
        if snaps and rng.random() > p_anchor:
            oppf = rng.choice(snaps[-12:])
        else:
            oppf = anchor
        opp.load_state_dict(T.load(oppf, map_location='cpu'))
        ls = rng.randrange(4)                          # learner seat rotates (gate uses all 4)
        store = []
        pols = [greedy(opp)] * 4
        pols[ls] = sample_pol(store)
        sim = Sim(pols, seed=seed * 1000 + g, quan=0, learner_seats=[ls], cnn=True)
        _, scores = sim.play()
        r = placement(scores, ls)                      # gate-matched terminal reward
        rows = [row + [r] for row in store]            # sparse terminal (single deal)
        out.append({'rows': rows, 'reward': r, 'opp': os.path.basename(oppf)})
    return out


class ResBNPV(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__()
        from models_explore import ResBNCNN
        self.net = ResBNCNN(channels=channels, blocks=blocks)
        self.vhead = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 9, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, obs, mask):
        f = self.net.body(self.net.stem(obs.float()))
        return self.net.foot(f) + torch.clamp(torch.log(mask.float()), -1e38, 1e38), self.vhead(f).squeeze(-1)


def ppo_update(model, sl_net, opt, rows, dev, clip, ent_c, epochs, beta):
    if not rows: return 0.0, 0.0, 0.0
    obs = torch.from_numpy(np.stack([d[0] for d in rows])).to(dev)
    mask = torch.from_numpy(np.stack([d[1] for d in rows])).to(dev)
    act = torch.tensor([d[2] for d in rows], device=dev)
    oldlp = torch.tensor([d[3] for d in rows], device=dev)
    ret = torch.tensor([d[4] for d in rows], device=dev, dtype=torch.float32)
    with torch.no_grad():
        _, vpred = model(obs, mask)
        slp = torch.softmax(sl_net({'is_training': False, 'obs': {'observation': obs, 'action_mask': mask}}), -1)
    adv = ret - vpred; adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    model.train(); kl_v = ent_v = 0.0
    for _ in range(epochs):
        lg, v = model(obs, mask); p = torch.softmax(lg, -1)
        lp = torch.log(p.gather(1, act[:, None]).squeeze(1) + 1e-9); ratio = torch.exp(lp - oldlp)
        pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        vl = F.mse_loss(v, ret); ent = -(p * torch.log(p + 1e-9)).sum(1).mean()
        kl = (p * (torch.log(p + 1e-9) - torch.log(slp + 1e-9))).sum(1).mean()
        loss = pl + 0.5 * vl - ent_c * ent + beta * kl
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        kl_v = float(kl); ent_v = float(ent)
    model.eval()
    return float(ret.mean()), kl_v, ent_v


def save_snapshot(model, blocks, snapdir, tag):
    """Write BN weights into the opponent pool + a fused resbn_fused .pkl (gateable) to snapdir."""
    from models_explore import ResBNCNN, fuse_resbn
    bn = ResBNCNN(channels=128, blocks=blocks); bn.load_state_dict(model.net.state_dict()); bn.eval()
    torch.save(bn.state_dict(), os.path.join(POOLDIR, f'm_{tag}.pkl'), _use_new_zipfile_serialization=False)
    fused = fuse_resbn(bn)
    fp = os.path.join(snapdir, f'snap_{tag}.pkl')
    torch.save(fused.state_dict(), fp, _use_new_zipfile_serialization=False)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='ckpt/aug/aug_128x40_s0.bn.pkl')
    ap.add_argument('--blocks', type=int, default=40)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--actors', type=int, default=24)
    ap.add_argument('--games-per-actor', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.008)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--beta-kl', type=float, default=0.4)
    ap.add_argument('--kl-decay', type=float, default=0.999)
    ap.add_argument('--beta-floor', type=float, default=0.05)
    ap.add_argument('--p-anchor', type=float, default=0.6)     # prob opponent = frozen aug_s0
    ap.add_argument('--snap-every', type=int, default=25)
    ap.add_argument('--pool-cap', type=int, default=20)
    ap.add_argument('--snapdir', default='ckpt/rl_online')
    ap.add_argument('--tag', default='rlon')
    ap.add_argument('--minutes', type=float, default=100000)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.snapdir, exist_ok=True)
    os.makedirs(POOLDIR, exist_ok=True)
    for f in glob.glob(POOLDIR + '/*.pkl'): os.remove(f)

    sl = torch.load(a.base, map_location='cpu')
    torch.save(sl, os.path.join(POOLDIR, '00_anchor.pkl'), _use_new_zipfile_serialization=False)
    from models_explore import ResBNCNN
    model = ResBNPV(blocks=a.blocks).to(dev); model.net.load_state_dict(sl)
    sl_net = ResBNCNN(channels=128, blocks=a.blocks).to(dev); sl_net.load_state_dict(sl); sl_net.eval()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    beta = a.beta_kl
    pool = mp.Pool(a.actors)
    t_start = time.time(); deadline = t_start + a.minutes * 60
    print(f"[rl_online] tag={a.tag} dev={dev} actors={a.actors} games/iter={a.actors*a.games_per_actor} "
          f"lr={a.lr} beta_kl={a.beta_kl} ent={a.ent} p_anchor={a.p_anchor}", flush=True)
    rew_hist = []
    for it in range(a.iters):
        if os.path.exists(STOP):
            print("[rl_online] STOP_RL present -> stopping", flush=True); break
        if time.time() > deadline:
            print("[rl_online] time cap reached -> stopping", flush=True); break
        t0 = time.time()
        torch.save(model.net.state_dict(), CUR_MAIN, _use_new_zipfile_serialization=False)
        jobs = [(it * 100000 + i * 131 + 7, a.games_per_actor, a.blocks, a.p_anchor) for i in range(a.actors)]
        results = pool.map(actor_play, jobs)
        rows = []; game_rews = []
        for games in results:
            for gm in games:
                rows += gm['rows']; game_rews.append(gm['reward'])
        r_mean, kl_v, ent_v = ppo_update(model, sl_net, opt, rows, dev, a.clip, a.ent, a.epochs, beta)
        beta = max(a.beta_floor, beta * a.kl_decay)
        gr = float(np.mean(game_rews)) if game_rews else 0.0
        rew_hist.append(gr); rew_hist = rew_hist[-50:]
        if (it + 1) % 5 == 0 or it == 0:
            print(f"it {it+1}/{a.iters} placement={gr:.3f} (ma50={np.mean(rew_hist):.3f}) "
                  f"kl={kl_v:.3f} ent={ent_v:.3f} beta={beta:.3f} rows={len(rows)} "
                  f"games={len(game_rews)} ({time.time()-t0:.0f}s)", flush=True)
        if (it + 1) % a.snap_every == 0:
            tag = f"{a.tag}_{it+1:05d}"
            fp = save_snapshot(model, a.blocks, a.snapdir, tag)
            print(f"  [snap] {fp} placement_ma50={np.mean(rew_hist):.3f}", flush=True)
            snaps = sorted(glob.glob(POOLDIR + '/m_*.pkl'))
            for f in snaps[:-(a.pool_cap)] if len(snaps) > a.pool_cap else []:
                os.remove(f)
    # final snapshot
    tag = f"{a.tag}_final"
    fp = save_snapshot(model, a.blocks, a.snapdir, tag)
    print(f"  [snap-final] {fp}", flush=True)
    pool.close(); print("DONE", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
