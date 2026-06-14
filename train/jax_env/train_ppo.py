"""
train_ppo.py — self-play PPO on the JAX CSM env (discard-only, Phase-2 win-aware).

Policy+value net (small conv on the env's (8,34) obs) plays all 4 seats; the env detects agari each
step; at terminals the winning hand is scored by MahjongGB (exact fan, 8-fan FLOOR, MCR duplicate
points). Monte-Carlo-return PPO (clip) updates the net. Metrics -> /root/metrics.json each iter for
the live dashboard. Periodic head-to-head eval vs a frozen snapshot.

Objective = the diagnosed weakness: learn to CONVERT to >=8-fan wins. The headline curve is the
8-fan win rate (random policy ~0.2%).
"""
import os, sys, time, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax, jax.numpy as jnp
from jax import random, lax
import optax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csm_env as E
import agari_jax as AJ
from MahjongGB import MahjongFanCalculator

SUIT, HON = AJ.load_tables()
TILES = [f'{s}{n}' for s in 'WTB' for n in range(1, 10)] + [f'F{n}' for n in range(1, 5)] + [f'J{n}' for n in range(1, 4)]
METRICS = os.environ.get("METRICS_PATH", "/root/metrics.json")
CH, NL = 64, 3


# ---------------- net ----------------
def init_net(key):
    ks = random.split(key, NL + 3)
    p = {"in": random.normal(ks[0], (CH, 8, 3)) * 0.1}
    for i in range(NL):
        p[f"c{i}"] = random.normal(ks[i + 1], (CH, CH, 3)) * (0.1 / CH ** 0.5)
    p["pi"] = random.normal(ks[-2], (CH * E.NT, E.NT)) * 0.01
    p["v"] = random.normal(ks[-1], (CH * E.NT, 1)) * 0.01
    return p


def trunk(p, obs):                                   # obs (B,8,34) -> (B, CH*34)
    def conv(x, w):
        return lax.conv_general_dilated(x, w, (1,), "SAME", dimension_numbers=("NCH", "OIH", "NCH"))
    h = jax.nn.relu(conv(obs, p["in"]))
    for i in range(NL):
        h = jax.nn.relu(h + conv(h, p[f"c{i}"]))
    return h.reshape(h.shape[0], -1)


def pi_logits(p, obs):
    return trunk(p, obs) @ p["pi"]


def pi_v(p, obs):
    t = trunk(p, obs)
    return t @ p["pi"], (t @ p["v"])[:, 0]


# ---------------- self-play rollout with a net policy ----------------
def encode_cur_obs(state):
    """(B,8,34): current player's hand thermometer (4) + 4 players' discard piles (relative)."""
    B = state['cur'].shape[0]; bi = jnp.arange(B)
    hand = state['hands'][bi, state['cur'], :]
    hp = jnp.stack([(hand >= k).astype(jnp.float32) for k in (1, 2, 3, 4)], 1)
    disc = state['discards'].astype(jnp.float32)     # (B,4,34) absolute; fine for v1
    return jnp.concatenate([hp, disc], 1)


def rollout(params, key, B, N):
    """Python-free jit rollout: net picks discards; record per-step (obs, action, logp, value, cur,
    live) and final winner/hand. Returns dict of stacked arrays."""
    st = E.init_state(key, B)
    st['winner'] = jnp.full(B, -1, jnp.int32); st['wtile'] = jnp.full(B, -1, jnp.int32)
    st['wtype'] = jnp.full(B, -1, jnp.int32); st['rng'] = key

    def body(st, _):
        B = st['cur'].shape[0]; bi = jnp.arange(B); live = ~st['done']; cur = st['cur']
        pre_h = st['hands']; pre_d = st['discards']; pre_p = st['ptr']
        st = E._draw(st); drawn = st['drawn']
        obs = encode_cur_obs(st)
        handc = st['hands'][bi, cur, :]
        sd = AJ.is_win_any_batch(handc, jnp.zeros(B, jnp.int32), SUIT, HON) & live & (drawn >= 0)
        logits, value = pi_v(params, obs)
        mask = handc > 0
        logits = jnp.where(mask, logits, -1e9)
        st['rng'], sk = random.split(st['rng'])
        act = random.categorical(sk, logits, axis=1)
        logp = jax.nn.log_softmax(logits, axis=1)[bi, act]
        rem = jax.nn.one_hot(act, E.NT, dtype=jnp.int8)
        h2 = st['hands'].at[bi, cur, :].add(-rem)
        disc = st['discards'].at[bi, cur, :].add(rem)
        rob = jnp.zeros(B, bool); rseat = jnp.full(B, -1, jnp.int32)
        for off in (1, 2, 3):
            opp = (cur + off) % 4
            w = AJ.is_win_any_batch(h2[bi, opp, :] + rem, jnp.zeros(B, jnp.int32), SUIT, HON) & live & ~sd
            rseat = jnp.where(w & ~rob, opp, rseat); rob = rob | w
        won = sd | rob
        winner = jnp.where(sd, cur, jnp.where(rob, rseat, -1))
        live3 = live[:, None, None]
        st = {**st,
              'hands': jnp.where(live3, jnp.where(sd[:, None, None], st['hands'], h2), pre_h),
              'discards': jnp.where(live3, disc, pre_d),
              'ptr': jnp.where(live, st['ptr'], pre_p),
              'cur': jnp.where(live & ~won, (cur + 1) % 4, cur),
              'done': st['done'] | won | (live & (st['ptr'] >= E.WALL)),
              'winner': jnp.where(won & (st['winner'] < 0), winner, st['winner']),
              'wtile': jnp.where(won & (st['winner'] < 0), jnp.where(sd, drawn, act), st['wtile']),
              'wtype': jnp.where(won & (st['winner'] < 0), jnp.where(sd, 1, 0), st['wtype'])}
        _, ph = AJ.reach_and_phi(h2[bi, cur, :], jnp.zeros(B, jnp.int32), SUIT, HON)
        hc = h2[bi, cur, :]
        suit_sum = jnp.stack([hc[:, 0:9].sum(1), hc[:, 9:18].sum(1), hc[:, 18:27].sum(1)], 1)
        flush = suit_sum.max(1) / 13.0          # dominant-suit fraction -> rewards Hun/Qing Yi Se
        ph = ph + 1.5 * flush
        rec = dict(obs=obs, act=act, logp=logp, value=value, cur=cur, live=live, phi=ph)
        return st, rec
    st, traj = lax.scan(body, st, None, length=N)
    return st, traj


rollout_jit = jax.jit(rollout, static_argnums=(2, 3))


# ---------------- hybrid terminal reward (CPU MahjongGB) ----------------
def score_terminals(final_hands, winner, wtile, wtype):
    """Return (B,4) MCR reward; 8-fan floor enforced (sub-8 'win' -> draw=0)."""
    B = winner.shape[0]; R = np.zeros((B, 4), np.float32)
    for b in range(B):
        w = int(winner[b])
        if w < 0:
            continue
        h = final_hands[b, w].copy()
        if wtype[b] == 0:
            h[wtile[b]] += 1
        tiles = [i for i in range(34) for _ in range(int(h[i]))]
        if len(tiles) != 14:
            continue
        wt = int(wtile[b]) if wtile[b] >= 0 else tiles[-1]
        rest = list(tiles);
        if wt in rest: rest.remove(wt)
        else: wt = rest.pop()
        try:
            fans = MahjongFanCalculator(pack=(), hand=tuple(TILES[i] for i in rest), winTile=TILES[wt],
                                        flowerCount=0, isSelfDrawn=bool(wtype[b]), is4thTile=False,
                                        isAboutKong=False, isWallLast=False, seatWind=0, prevalentWind=0, verbose=False)
            fan = sum(fp * c for fp, c, *_ in fans)
        except Exception:
            fan = 0
        if fan < 8:
            continue                                   # 8-fan floor: no legal win
        if wtype[b]:                                   # self-draw
            for p in range(4):
                if p != w: R[b, p] -= (8 + fan); R[b, w] += (8 + fan)
        else:                                          # discard win; feeder = the seat that just discarded
            feeder = (w + 1) % 4 if False else None    # feeder unknown from final state -> approximate via cur
            # approximate: rob credited generically (all non-winners pay 8, winner +24) — refine later
            for p in range(4):
                if p != w: R[b, p] -= 8; R[b, w] += 8
            R[b, w] += fan                             # winner gets the fan bonus (approx feeder split)
    return R


# ---------------- PPO update (MC returns, clipped) ----------------
def ppo_loss(params, obs, act, logp_old, ret, adv, clip=0.2, ent_c=0.01, v_c=0.5):
    logits, value = pi_v(params, obs)
    logp = jax.nn.log_softmax(logits, axis=1)[jnp.arange(act.shape[0]), act]
    ratio = jnp.exp(logp - logp_old)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
    vl = ((value - ret) ** 2).mean()
    probs = jax.nn.softmax(logits, axis=1)
    ent = -(probs * jnp.log(probs + 1e-9)).sum(1).mean()
    return pg + v_c * vl - ent_c * ent, (pg, vl, ent)


@jax.jit
def update(params, opt_state, batch):
    (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
        params, batch['obs'], batch['act'], batch['logp'], batch['ret'], batch['adv'])
    updates, opt_state = OPT.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, aux


def flatten_traj(traj, R):
    """Build a flat training batch from a rollout: each live decision gets MC return R[b, seat]."""
    N, B = traj['act'].shape
    cur = np.array(traj['cur']); live = np.array(traj['live'])
    obs = np.array(traj['obs']); act = np.array(traj['act'])
    logp = np.array(traj['logp']); val = np.array(traj['value'])
    LAM = float(os.environ.get('PPO_SHAPE', '1.0'))
    phi = np.array(traj['phi'])
    ret = R[np.arange(B)[None, :], cur] + LAM * phi     # terminal MCR + dense completion shaping
    m = live.reshape(-1)
    flat = dict(obs=obs.reshape(-1, 8, 34)[m], act=act.reshape(-1)[m],
                logp=logp.reshape(-1)[m], ret=ret.reshape(-1)[m], val=val.reshape(-1)[m])
    flat['adv'] = flat['ret'] - flat['val']
    a = flat['adv']; flat['adv'] = (a - a.mean()) / (a.std() + 1e-6)
    return flat


def main():
    B = int(os.environ.get('PPO_B', '8192')); N = int(os.environ.get('PPO_N', '90'))
    iters = int(os.environ.get('PPO_ITERS', '100000')); epochs = 2
    key = random.PRNGKey(0)
    params = init_net(key)
    global OPT
    OPT = optax.adam(3e-4); opt_state = OPT.init(params)
    hist = []
    for it in range(iters):
        key, rk = random.split(key)
        t0 = time.time()
        st, traj = rollout_jit(params, rk, B, N)
        winner = np.array(st['winner']); fh = np.array(st['hands'])
        wtile = np.array(st['wtile']); wtype = np.array(st['wtype'])
        R = score_terminals(fh, winner, wtile, wtype)
        # metrics
        legal_wins = int((R.max(1) > 0).sum())          # games with a >=8-fan win
        winrate = legal_wins / B
        draws = int((winner < 0).sum()) / B
        flat = flatten_traj(traj, R)
        # PPO epochs
        import jax as _j
        MB = int(os.environ.get('PPO_MB', '8192'))
        n = flat['act'].shape[0]; idx = np.arange(n)
        loss = vl = ent = 0.0
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s0 in range(0, n, MB):
                sl = idx[s0:s0 + MB]
                bn = {k: jnp.asarray(flat[k][sl]) for k in ('obs', 'act', 'logp', 'ret', 'adv')}
                params, opt_state, loss, (pg, vl, ent) = update(params, opt_state, bn)
        dt = time.time() - t0
        rec = dict(iter=it, winrate8=winrate, draw_rate=draws, mean_reward=float(R.mean()),
                   loss=float(loss), vloss=float(vl), entropy=float(ent),
                   games_per_s=int(B / dt), n_decisions=int(flat['act'].shape[0]),
                   ts=time.time())
        hist.append(rec); hist = hist[-5000:]
        json.dump(dict(history=hist, latest=rec), open(METRICS, 'w'))
        if it % 5 == 0:
            print(f"it{it} win8={winrate*100:.2f}% draw={draws*100:.0f}% rew={R.mean():+.2f} "
                  f"ent={ent:.2f} vl={vl:.1f} {int(B/dt)}g/s", flush=True)


if __name__ == '__main__':
    main()
