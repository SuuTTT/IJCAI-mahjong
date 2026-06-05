"""
rl_league.py — LEAGUE training (main + exploiter + PFSP) for the CNN, the research's fix
for the non-transitivity "parity trap" that pool+KL (rl_actors.py) hit (39-39 tie vs base).

Two learners, one GPU process:
  * MAIN     — PPO + KL-to-SL leash. The deploy candidate. Trained vs a PFSP mixture of the
               opponent pool (frozen SL + main snapshots + exploiter snapshots), oversampling
               whichever opponents the main currently LOSES to (PFSP weight (1-winrate)^p).
  * EXPLOITER — PPO, NO leash. Trained only to beat the *current frozen main*. When it clears a
               win-rate threshold vs the main, its snapshot is added to the main's opponent pool
               (forcing the main to patch that hole) and the exploiter resets from the current main.

Pool dir /tmp/leaguepool: 00_sl.pkl (frozen SL anchor), m_XX.pkl (main snaps), e_XX.pkl (exp snaps).
Reward = terminal (seat0+seat2 score)/48, as in rl_actors (proven stable).

  python3 rl_league.py --base arch_ck/explore/resbn40.pkl --blocks 40 --iters 700 \
      --main-actors 16 --exp-actors 6 --games-per-actor 3 \
      --out arch_ck/explore/resbn40_league.pkl
"""
import os, sys, json, argparse, time, random, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import multiprocessing as mp

POOLDIR = '/tmp/leaguepool'
CUR_MAIN = '/tmp/lg_main.pkl'
CUR_EXP = '/tmp/lg_exp.pkl'
PHIPATH = os.environ.get('PHI_PATH', '')   # if set, dense potential-based shaping via Φ
GAMMA = float(os.environ.get('PHI_GAMMA', '0.999'))

# ---------------- actor (CPU worker) ----------------
def actor_play(arg):
    # role: 'main' -> cur=MAIN(0,2) vs opp from pool(1,3); returns rows + opp_id + main_won
    #       'exp'  -> cur=EXP(0,2)  vs MAIN frozen(1,3);   returns rows + exp_won
    # DENSE: per-seat store, return G_t = Σ γ^{k-t}(r_k + γΦ(s_{k+1})-Φ(s_k)) (potential shaping).
    seed, n_games, blocks, role, opp_choices = arg
    import torch as T
    T.set_num_threads(1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models_explore import ResBNCNN
    from sim_cnn import Sim
    rng = random.Random(seed)
    curpath = CUR_MAIN if role == 'main' else CUR_EXP
    cur = ResBNCNN(channels=128, blocks=blocks); cur.load_state_dict(T.load(curpath, map_location='cpu')); cur.eval()
    opp = ResBNCNN(channels=128, blocks=blocks); opp.eval()
    phi = None
    if PHIPATH and os.path.exists(PHIPATH):
        from phi_reward import PhiNet
        phi = PhiNet(); phi.load_state_dict(T.load(PHIPATH, map_location='cpu')); phi.eval()

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

    def shaped_rows(store, term_r):
        """Build per-row Monte-Carlo returns with potential-based shaping (or flat term_r if no Φ)."""
        if not store:
            return []
        if phi is None:
            return [row + [term_r] for row in store]
        with T.no_grad():
            ob = T.from_numpy(np.stack([row[0] for row in store]).astype(np.float32))
            phis = phi(ob).numpy().tolist()                # Φ(s_t) for each decision state
        n = len(store)
        rew = [0.0] * n; rew[-1] = term_r                  # sparse terminal team reward
        # shaping F_t = γΦ(s_{t+1}) - Φ(s_t); Φ(terminal)=0
        shp = [GAMMA * (phis[t + 1] if t + 1 < n else 0.0) - phis[t] for t in range(n)]
        G = [0.0] * n; acc = 0.0
        for t in range(n - 1, -1, -1):
            acc = (rew[t] + shp[t]) + GAMMA * acc
            G[t] = acc
        return [store[t] + [G[t]] for t in range(n)]

    out = []
    for g in range(n_games):
        if role == 'main':
            oppf = _pick(opp_choices, rng)        # PFSP-weighted pool file
            opp.load_state_dict(T.load(oppf, map_location='cpu'))
        else:
            oppf = CUR_MAIN                        # exploiter always fights the current main
            opp.load_state_dict(T.load(CUR_MAIN, map_location='cpu'))
        s0, s2 = [], []                            # per-seat stores (temporal order preserved)
        sim = Sim([sample_pol(s0), greedy(opp), sample_pol(s2), greedy(opp)],
                  seed=seed * 1000 + g, quan=0, learner_seats=[0, 2], cnn=True)
        sim.play()
        r = (sim.scores[0] + sim.scores[2]) / 48.0
        won = (sim.scores[0] + sim.scores[2]) > (sim.scores[1] + sim.scores[3])
        rows = shaped_rows(s0, r) + shaped_rows(s2, r)
        out.append({'rows': rows, 'opp': os.path.basename(oppf), 'won': won, 'role': role})
    return out

def _pick(choices, rng):
    # choices = [(path, weight), ...]; weighted sample
    tot = sum(w for _, w in choices); x = rng.random() * tot; c = 0.0
    for p, w in choices:
        c += w
        if x <= c: return p
    return choices[-1][0]


class ResBNPV(nn.Module):
    def __init__(self, channels=128, blocks=40):
        super().__init__(); from models_explore import ResBNCNN
        self.net = ResBNCNN(channels=channels, blocks=blocks)
        self.vhead = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 9, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, obs, mask):
        f = self.net.body(self.net.stem(obs.float()))
        return self.net.foot(f) + torch.clamp(torch.log(mask.float()), -1e38, 1e38), self.vhead(f).squeeze(-1)


def ppo_update(model, sl_net, opt, rows, dev, clip, ent_c, epochs, beta, use_kl):
    if not rows: return 0.0, 0.0
    obs = torch.from_numpy(np.stack([d[0] for d in rows])).to(dev)
    mask = torch.from_numpy(np.stack([d[1] for d in rows])).to(dev)
    act = torch.tensor([d[2] for d in rows], device=dev)
    oldlp = torch.tensor([d[3] for d in rows], device=dev)
    ret = torch.tensor([d[4] for d in rows], device=dev)
    with torch.no_grad():
        _, vpred = model(obs, mask)
        if use_kl:
            slp = torch.softmax(sl_net({'is_training': False, 'obs': {'observation': obs, 'action_mask': mask}}), -1)
    adv = ret - vpred; adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    model.train(); kl_v = 0.0
    for _ in range(epochs):
        lg, v = model(obs, mask); p = torch.softmax(lg, -1)
        lp = torch.log(p.gather(1, act[:, None]).squeeze(1) + 1e-9); ratio = torch.exp(lp - oldlp)
        pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        vl = F.mse_loss(v, ret); ent = -(p * torch.log(p + 1e-9)).sum(1).mean()
        loss = pl + 0.5 * vl - ent_c * ent
        if use_kl:
            kl = (p * (torch.log(p + 1e-9) - torch.log(slp + 1e-9))).sum(1).mean()
            loss = loss + beta * kl; kl_v = float(kl)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    return float(ret.mean()), kl_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--blocks', type=int, default=40)
    ap.add_argument('--iters', type=int, default=700)
    ap.add_argument('--main-actors', type=int, default=16); ap.add_argument('--exp-actors', type=int, default=6)
    ap.add_argument('--games-per-actor', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-5); ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01); ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--beta-kl', type=float, default=0.4); ap.add_argument('--kl-decay', type=float, default=0.997)
    ap.add_argument('--pfsp-p', type=float, default=2.0)        # PFSP weight exponent
    ap.add_argument('--exp-promote', type=float, default=0.58)  # exploiter winrate vs main to snapshot
    ap.add_argument('--pool-cap', type=int, default=24)
    ap.add_argument('--snap-every', type=int, default=10); ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--gauntlet-games', type=int, default=0)   # >0: gate --out on diverse-gauntlet net (#23)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(POOLDIR, exist_ok=True)
    for f in glob.glob(POOLDIR + '/*.pkl'): os.remove(f)
    sl = torch.load(a.base, map_location='cpu')
    torch.save(sl, os.path.join(POOLDIR, '00_sl.pkl'), _use_new_zipfile_serialization=False)
    from models_explore import ResBNCNN
    main_m = ResBNPV(blocks=a.blocks).to(dev); main_m.net.load_state_dict(sl)
    exp_m = ResBNPV(blocks=a.blocks).to(dev); exp_m.net.load_state_dict(sl)
    sl_net = ResBNCNN(channels=128, blocks=a.blocks).to(dev); sl_net.load_state_dict(sl); sl_net.eval()
    opt_m = torch.optim.Adam(main_m.parameters(), lr=a.lr)
    opt_e = torch.optim.Adam(exp_m.parameters(), lr=a.lr)
    beta = a.beta_kl; best_gnet = -10**9
    if a.gauntlet_games > 0:                                   # seed the bar at the START model's strength
        from gauntlet_eval import gauntlet_net
        _cpu = ResBNCNN(channels=128, blocks=a.blocks); _cpu.load_state_dict(main_m.net.state_dict()); _cpu.eval()
        best_gnet = gauntlet_net(_cpu, n_games=a.gauntlet_games)
        print(f"  [gauntlet] START net={best_gnet:+d} (promote only if beaten)", flush=True)
    # winrate tracking for PFSP: {pool_file: [w, n]} (main's results vs that opponent), decayed
    wr = {}
    snap_m = snap_e = 1
    pool = mp.Pool(a.main_actors + a.exp_actors)

    def pfsp_choices():
        files = sorted(glob.glob(POOLDIR + '/*.pkl'))
        ch = []
        for f in files:
            b = os.path.basename(f); w, n = wr.get(b, (0.0, 0.0))
            p = (w / n) if n > 0 else 0.5                 # main winrate vs this opp
            weight = (1.0 - p) ** a.pfsp_p + 0.05         # oversample hard opps, small floor
            ch.append((f, weight))
        return ch

    exp_recent = []  # rolling exploiter wins vs main (1/0)
    for it in range(a.iters):
        t0 = time.time()
        torch.save(main_m.net.state_dict(), CUR_MAIN, _use_new_zipfile_serialization=False)
        torch.save(exp_m.net.state_dict(), CUR_EXP, _use_new_zipfile_serialization=False)
        ch = pfsp_choices()
        jobs = [(it * 1000 + i, a.games_per_actor, a.blocks, 'main', ch) for i in range(a.main_actors)]
        jobs += [(it * 1000 + 500 + i, a.games_per_actor, a.blocks, 'exp', None) for i in range(a.exp_actors)]
        results = pool.map(actor_play, jobs)
        main_rows, exp_rows = [], []
        for games in results:
            for gm in games:
                if gm['role'] == 'main':
                    main_rows += gm['rows']
                    b = gm['opp']; w, n = wr.get(b, (0.0, 0.0))
                    wr[b] = (w * 0.97 + (1.0 if gm['won'] else 0.0), n * 0.97 + 1.0)  # decayed winrate
                else:
                    exp_rows += gm['rows']
                    exp_recent.append(1 if gm['won'] else 0)
        exp_recent = exp_recent[-60:]
        r_m, kl_m = ppo_update(main_m, sl_net, opt_m, main_rows, dev, a.clip, a.ent, a.epochs, beta, use_kl=True)
        r_e, _ = ppo_update(exp_m, sl_net, opt_e, exp_rows, dev, a.clip, a.ent, a.epochs, 0.0, use_kl=False)
        beta *= a.kl_decay
        ewr = (sum(exp_recent) / len(exp_recent)) if exp_recent else 0.0
        print(f"it {it+1}/{a.iters} main_r={r_m:+.3f} kl={kl_m:.3f} beta={beta:.3f} "
              f"exp_r={r_e:+.3f} exp_wr_vs_main={ewr:.2f} m={len(main_rows)} e={len(exp_rows)} ({time.time()-t0:.0f}s)", flush=True)

        if (it + 1) % a.snap_every == 0:
            torch.save(main_m.net.state_dict(), os.path.join(POOLDIR, f'm_{snap_m:02d}.pkl'), _use_new_zipfile_serialization=False); snap_m += 1
        # promote exploiter when it reliably beats the main, then reset it from the current main
        if ewr >= a.exp_promote and len(exp_recent) >= 40:
            torch.save(exp_m.net.state_dict(), os.path.join(POOLDIR, f'e_{snap_e:02d}.pkl'), _use_new_zipfile_serialization=False); snap_e += 1
            exp_m.net.load_state_dict(main_m.net.state_dict()); exp_recent = []
            print(f"  [promote] exploiter e_{snap_e-1:02d} added to pool (wr {ewr:.2f}); exploiter reset to main", flush=True)
        # cap pool (keep 00_sl always)
        snaps = sorted(glob.glob(POOLDIR + '/[me]_*.pkl'))
        for f in snaps[:-(a.pool_cap)] if len(snaps) > a.pool_cap else []:
            os.remove(f); wr.pop(os.path.basename(f), None)
        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            if a.gauntlet_games > 0:                                   # #23: gate on diverse-gauntlet net
                from gauntlet_eval import gauntlet_net
                cpu = ResBNCNN(channels=128, blocks=a.blocks); cpu.load_state_dict(main_m.net.state_dict()); cpu.eval()
                gnet = gauntlet_net(cpu, n_games=a.gauntlet_games)
                if gnet > best_gnet:
                    best_gnet = gnet
                    torch.save(main_m.net.state_dict(), a.out, _use_new_zipfile_serialization=False)
                    print(f"  [gauntlet] net={gnet:+d} NEW BEST -> saved {a.out}", flush=True)
                else:
                    print(f"  [gauntlet] net={gnet:+d} (best {best_gnet:+d}, not promoted)", flush=True)
            else:
                torch.save(main_m.net.state_dict(), a.out, _use_new_zipfile_serialization=False); print(f"  saved -> {a.out}", flush=True)
    pool.close(); print("DONE", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
