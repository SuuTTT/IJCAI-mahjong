"""
train_ppo_ws.py — WARM-STARTED self-play PPO on the JAX CSM env.

The policy is the deploy SL net (ResFused-40/128, lad_chunjiandu) loaded from cnn_lad_chunjiandu.npz —
NOT a from-scratch conv. This is the Tjong "SL first, then RL" fix for the win8=0 sparse-reward wall:
the policy starts COMPETENT (it already knows which tile to discard) and RL only has to push it toward
>=8-fan conversion. The 235-action SL head is restricted to its 34 Play/discard logits (indices 2..36);
a fresh value head reads the penultimate features.

Obs = the deploy 38-plane CAIEST encoding (obs38, byte-exact vs feature.py). Terminal reward = MahjongGB
exact fan with the 8-fan FLOOR (sub-8 'win' -> draw), plus dense completion shaping (phi). Metrics ->
/root/metrics_ws.json. Headline = the 8-fan win rate vs the frozen warm-start baseline.
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
import obs38 as O38
import resnet_jax as RJ
from MahjongGB import MahjongFanCalculator

SUIT, HON = AJ.load_tables()
TILES = [f'{s}{n}' for s in 'WTB' for n in range(1, 10)] + [f'F{n}' for n in range(1, 5)] + [f'J{n}' for n in range(1, 4)]
METRICS = os.environ.get("METRICS_PATH", "/root/metrics_ws.json")
NPZ = os.environ.get("WS_NPZ", "cnn_lad_chunjiandu.npz")
PLAY0 = 2          # OFFSET_ACT['Play']: discard logits live at full_logits[:, 2:36]
NB = None          # number of residual blocks, set in build_params


# ---------------- warm-started net ----------------
def build_params(key):
    """SL ResFused params (trainable) + fresh value head reading the 512-d penultimate features."""
    global NB
    sl = RJ.load_params(NPZ)
    NB = int(sl.pop('_blocks'))
    p = {k: v for k, v in sl.items()}              # all float32 jax arrays
    p['v.w'] = random.normal(key, (512, 1)) * 0.01
    p['v.b'] = jnp.zeros((1,))
    return p


def pi_v(p, obs):
    """obs (B,38,4,9) -> (discard_logits (B,34), value (B,))."""
    full, h = RJ.forward_feats(p, obs, NB)
    return full[:, PLAY0:PLAY0 + 34], (h @ p['v.w'] + p['v.b'])[:, 0]


# ---------------- self-play rollout (net policy, warm-started) ----------------
def rollout(params, key, B, N):
    st = E.init_state(key, B)
    st['winner'] = jnp.full(B, -1, jnp.int32); st['wtile'] = jnp.full(B, -1, jnp.int32)
    st['wtype'] = jnp.full(B, -1, jnp.int32); st['rng'] = key

    def body(st, _):
        B = st['cur'].shape[0]; bi = jnp.arange(B); live = ~st['done']; cur = st['cur']
        pre_h = st['hands']; pre_d = st['discards']; pre_p = st['ptr']
        st = E._draw(st); drawn = st['drawn']
        obs = O38.encode_obs38(st['hands'], st['discards'], cur)     # (B,38,4,9), post-draw hand
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
        flush = suit_sum.max(1) / 13.0
        ph = ph + 1.5 * flush
        rec = dict(obs=obs, act=act, logp=logp, value=value, cur=cur, live=live, phi=ph)
        return st, rec
    st, traj = lax.scan(body, st, None, length=N)
    return st, traj


rollout_jit = jax.jit(rollout, static_argnums=(2, 3))


# ---------------- terminal reward (CPU MahjongGB, 8-fan floor) ----------------
def score_terminals(final_hands, winner, wtile, wtype):
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
        rest = list(tiles)
        if wt in rest: rest.remove(wt)
        else: wt = rest.pop()
        try:
            fans = MahjongFanCalculator(pack=(), hand=tuple(TILES[i] for i in rest), winTile=TILES[wt],
                                        flowerCount=0, isSelfDrawn=bool(wtype[b]), is4thTile=False,
                                        isAboutKong=False, isWallLast=False, seatWind=0, prevalentWind=0, verbose=True)
            fan = sum(fp * c for fp, c, *_ in fans)   # verbose=True -> (fanValue, count, cn, en)
        except Exception:
            fan = 0
        if fan < 8:
            continue
        if wtype[b]:
            for p in range(4):
                if p != w: R[b, p] -= (8 + fan); R[b, w] += (8 + fan)
        else:
            for p in range(4):
                if p != w: R[b, p] -= 8; R[b, w] += 8
            R[b, w] += fan
    return R


# ---------------- PPO update ----------------
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
    N, B = traj['act'].shape
    cur = np.array(traj['cur']); live = np.array(traj['live'])
    obs = np.array(traj['obs']); act = np.array(traj['act'])
    logp = np.array(traj['logp']); val = np.array(traj['value'])
    LAM = float(os.environ.get('PPO_SHAPE', '1.0'))
    phi = np.array(traj['phi'])
    ret = R[np.arange(B)[None, :], cur] + LAM * phi
    m = live.reshape(-1)
    flat = dict(obs=obs.reshape(-1, 38, 4, 9)[m], act=act.reshape(-1)[m],
                logp=logp.reshape(-1)[m], ret=ret.reshape(-1)[m], val=val.reshape(-1)[m])
    flat['adv'] = flat['ret'] - flat['val']
    a = flat['adv']; flat['adv'] = (a - a.mean()) / (a.std() + 1e-6)
    return flat


def save_ckpt(params, path):
    np.savez(path, **{k: np.array(v) for k, v in params.items()}, meta_blocks=np.array([NB]))


def main():
    B = int(os.environ.get('PPO_B', '4096')); N = int(os.environ.get('PPO_N', '90'))
    iters = int(os.environ.get('PPO_ITERS', '100000')); epochs = 2
    lr = float(os.environ.get('PPO_LR', '3e-5'))            # low: fine-tune a competent net, don't wreck it
    key = random.PRNGKey(0)
    params = build_params(key)
    global OPT
    OPT = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = OPT.init(params)
    hist = []
    for it in range(iters):
        key, rk = random.split(key)
        t0 = time.time()
        st, traj = rollout_jit(params, rk, B, N)
        winner = np.array(st['winner']); fh = np.array(st['hands'])
        wtile = np.array(st['wtile']); wtype = np.array(st['wtype'])
        R = score_terminals(fh, winner, wtile, wtype)
        legal_wins = int((R.max(1) > 0).sum())
        winrate = legal_wins / B
        draws = int((winner < 0).sum()) / B
        flat = flatten_traj(traj, R)
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
                   games_per_s=int(B / dt), n_decisions=int(flat['act'].shape[0]), ts=time.time())
        hist.append(rec); hist = hist[-5000:]
        json.dump(dict(history=hist, latest=rec), open(METRICS, 'w'))
        if it % 5 == 0:
            print(f"it{it} win8={winrate*100:.2f}% draw={draws*100:.0f}% rew={R.mean():+.2f} "
                  f"ent={ent:.2f} vl={vl:.1f} {int(B/dt)}g/s", flush=True)
        if it % 200 == 0 and it > 0:
            save_ckpt(params, os.environ.get('CKPT', '/root/ws_ckpt.npz'))


if __name__ == '__main__':
    main()
