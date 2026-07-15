"""PPO self-play trainer for the Ludus MCR RL environment (``mahjong.rl_env``).

CPU-only by design: the Mahjong engine is pure Python, so rollouts are
parallelised across processes (``multiprocessing``, spawn) while a small
JAX/flax actor-critic CNN over (38,4,9) does the learning.  Rollout workers run
a **pure-NumPy** copy of the network (no JAX in workers -> trivially picklable,
no fork/JAX hazards); the NumPy forward is parity-checked against the flax model
at startup.

Learner controls seat 0; opponents (seats 1-3) are a mix of frozen self-play
snapshots and easy baselines (EfficiencyBot / RandomLegalBot), the self-play
fraction ramping up over training so there is a learning signal from the start.

8-fan wins are RARE from scratch, so expect SLOW learning; the visible early
signals are usually (a) mean reward trending up toward 0 (fewer deal-ins) and
(b) entropy falling, before win-rate climbs.

Usage:
    JAX_PLATFORMS=cpu python -m baselines.mahjong_ppo \
        --updates 200 --workers 24 --games-per-update 256 --lr 3e-4 \
        --eval-every 10 --out /root/ludus_train/mahjong --resume
"""

import os
# Pin BLAS to a single thread BEFORE importing numpy: the per-decision matmuls
# are tiny, so multi-threaded BLAS across many rollout workers causes severe
# oversubscription (measured ~100x slowdown).  This runs at module top in both
# the main process and every spawned worker (which re-imports this module).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# NumPy inference (used inside rollout/eval workers -- no JAX here)
# ----------------------------------------------------------------------------
CONV_CH = 64
HID = 256


def _conv_same(x, w, b):
    """SAME-padded stride-1 conv, NHWC input, kernel (kh,kw,Cin,Cout). Mirrors
    flax.linen.Conv defaults exactly (odd kernel -> symmetric pad)."""
    N, H, W, Cin = x.shape
    kh, kw, _, Cout = w.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    out = np.zeros((N, H, W, Cout), dtype=np.float32)
    for di in range(kh):
        for dj in range(kw):
            patch = xp[:, di:di + H, dj:dj + W, :].reshape(N * H * W, Cin)
            out += (patch @ w[di, dj]).reshape(N, H, W, Cout)
    return out + b


def forward_np(P, obs):
    """(38,4,9) obs -> (logits (235,), value scalar) via the NumPy mirror."""
    x = np.transpose(obs, (1, 2, 0))[None].astype(np.float32)     # (1,4,9,38)
    x = np.maximum(_conv_same(x, P["c0w"], P["c0b"]), 0.0)
    x = np.maximum(_conv_same(x, P["c1w"], P["c1b"]), 0.0)
    x = x.reshape(1, -1)
    x = np.maximum(x @ P["d0w"] + P["d0b"], 0.0)
    logits = (x @ P["piw"] + P["pib"])[0]
    value = float((x @ P["vw"] + P["vb"])[0, 0])
    return logits, value


def _masked_logp(logits, mask):
    """log-softmax over legal actions only."""
    ml = np.where(mask, logits, -1e30)
    ml = ml - ml.max()
    ex = np.exp(ml) * mask
    Z = ex.sum()
    return ml - np.log(Z + 1e-30), ex / (Z + 1e-30)


def act_sample(P, obs, mask, rng):
    logits, value = forward_np(P, obs)
    logp, p = _masked_logp(logits, mask)
    a = int(rng.choice(len(p), p=p))
    return a, float(logp[a]), value


def act_greedy(P, obs, mask):
    logits, _ = forward_np(P, obs)
    return int(np.argmax(np.where(mask, logits, -1e30)))


# ----------------------------------------------------------------------------
# Rollout / eval workers (spawned; pure NumPy + the engine)
# ----------------------------------------------------------------------------
def _make_opp(kind, opp_params, seed):
    # imported lazily so workers don't need JAX
    from mahjong.rl_env import PolicyAgent
    from mahjong.bots import EfficiencyBot, RandomLegalBot
    if kind == "eff":
        return EfficiencyBot(seed)
    if kind == "rand":
        return RandomLegalBot(seed)
    rng = np.random.default_rng(seed)
    return PolicyAgent(lambda o, m: act_sample(opp_params, o, m, rng)[0])


def rollout_chunk(task):
    """Play a chunk of games; return per-game trajectory dicts."""
    sys.path.insert(0, "/root/ludus")
    from mahjong.rl_env import MahjongEnv
    P = task["params"]
    OP = task["opp_params"]
    out = []
    for seed, opp_kinds in task["games"]:
        rng = np.random.default_rng(seed ^ 0x9E3779B9)
        env = MahjongEnv(reward_shaping=task["shaping"])
        opps = [_make_opp(opp_kinds[i], OP, seed * 4 + i + 1) for i in range(3)]
        obs, mask = env.reset(opps, seed=seed)
        O, M, A, LP, V, R = [], [], [], [], [], []
        while True:
            a, lp, v = act_sample(P, obs, mask, rng)
            O.append(obs); M.append(mask); A.append(a)
            LP.append(lp); V.append(v)
            obs, mask, r, done, info = env.step(a)
            R.append(r)
            if done:
                break
        out.append(dict(
            obs=np.asarray(O, np.float32), mask=np.asarray(M, bool),
            act=np.asarray(A, np.int32), logp=np.asarray(LP, np.float32),
            val=np.asarray(V, np.float32), rew=np.asarray(R, np.float32),
            ending=info["ending"], winner=info["winner"]))
    return out


def eval_chunk(task):
    """Greedy learner vs fixed opponents; return (win, reward, ending) list."""
    sys.path.insert(0, "/root/ludus")
    from mahjong.rl_env import MahjongEnv
    P = task["params"]
    res = []
    for seed in task["seeds"]:
        env = MahjongEnv()
        opps = [_make_opp(task["opp"], None, seed * 4 + i + 1) for i in range(3)]
        obs, mask = env.reset(opps, seed=seed)
        while True:
            a = act_greedy(P, obs, mask)
            obs, mask, r, done, info = env.step(a)
            if done:
                break
        res.append((1 if info["winner"] == 0 else 0, r, info["ending"]))
    return res


# ----------------------------------------------------------------------------
# JAX / flax model + PPO update (main process only)
# ----------------------------------------------------------------------------
def build_jax():
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    class ActorCritic(nn.Module):
        @nn.compact
        def __call__(self, x):                       # x: (N,38,4,9)
            x = jnp.transpose(x, (0, 2, 3, 1))       # NHWC (N,4,9,38)
            x = nn.relu(nn.Conv(CONV_CH, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(CONV_CH, (3, 3), padding="SAME")(x))
            x = x.reshape((x.shape[0], -1))
            x = nn.relu(nn.Dense(HID)(x))
            logits = nn.Dense(235)(x)
            v = nn.Dense(1)(x)[..., 0]
            return logits, v

    return jax, jnp, nn, ActorCritic


def params_to_np(params):
    p = params["params"]
    g = lambda k, s: np.asarray(p[k][s], np.float32)
    return {
        "c0w": g("Conv_0", "kernel"), "c0b": g("Conv_0", "bias"),
        "c1w": g("Conv_1", "kernel"), "c1b": g("Conv_1", "bias"),
        "d0w": g("Dense_0", "kernel"), "d0b": g("Dense_0", "bias"),
        "piw": g("Dense_1", "kernel"), "pib": g("Dense_1", "bias"),
        "vw": g("Dense_2", "kernel"), "vb": g("Dense_2", "bias"),
    }


def compute_gae(traj, gamma, lam):
    r, v = traj["rew"], traj["val"]
    T = len(r)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nextv = v[t + 1] if t < T - 1 else 0.0      # episode ends at T-1
        nonterm = 0.0 if t == T - 1 else 1.0
        delta = r[t] + gamma * nextv * nonterm - v[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--games-per-update", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--shaping", type=float, default=0.0)
    ap.add_argument("--selfplay-warmup", type=int, default=40,
                    help="updates over which self-play fraction ramps 0->1")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=64)
    ap.add_argument("--out", type=str, default="/root/ludus_train/mahjong")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.makedirs(args.out, exist_ok=True)
    ctx = mp.get_context("spawn")

    jax, jnp, nn, ActorCritic = build_jax()
    import optax
    from flax.training import train_state
    from flax import serialization

    model = ActorCritic()
    key = jax.random.PRNGKey(args.seed)
    dummy = jnp.zeros((1, 38, 4, 9), jnp.float32)
    params = model.init(key, dummy)

    # --- NumPy-vs-JAX parity check ---
    rng0 = np.random.default_rng(0)
    ob = rng0.standard_normal((38, 4, 9)).astype(np.float32)
    jl, jv = model.apply(params, ob[None])
    Pnp = params_to_np(params)
    nl, nvv = forward_np(Pnp, ob)
    dl = float(np.max(np.abs(np.asarray(jl)[0] - nl)))
    dv = float(abs(float(jv[0]) - nvv))
    print(f"[parity] max|logit diff|={dl:.2e}  |value diff|={dv:.2e}", flush=True)
    assert dl < 1e-3 and dv < 1e-3, "NumPy forward does not match flax model!"

    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    start_update = 0
    best_metric = -1e9
    last_path = os.path.join(args.out, "last.msgpack")
    meta_path = os.path.join(args.out, "last.json")
    if args.resume and os.path.exists(last_path):
        with open(last_path, "rb") as f:
            state = state.replace(params=serialization.from_bytes(state.params, f.read()))
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            start_update = meta.get("update", 0)
            best_metric = meta.get("best_metric", -1e9)
        print(f"[resume] loaded {last_path} at update {start_update}", flush=True)

    def save(path, prms):
        with open(path, "wb") as f:
            f.write(serialization.to_bytes(prms))

    @jax.jit
    def ppo_update(state, obs, mask, act, logp_old, adv, ret):
        def loss_fn(params):
            logits, value = model.apply(params, obs)
            logits = jnp.where(mask, logits, -1e30)
            logp_all = jax.nn.log_softmax(logits, axis=-1)
            logp = jnp.take_along_axis(logp_all, act[:, None], axis=-1)[:, 0]
            ratio = jnp.exp(logp - logp_old)
            pg1 = ratio * adv
            pg2 = jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * adv
            pg_loss = -jnp.mean(jnp.minimum(pg1, pg2))
            v_loss = jnp.mean((value - ret) ** 2)
            p = jnp.exp(logp_all) * mask
            ent = -jnp.mean(jnp.sum(p * jnp.where(mask, logp_all, 0.0), axis=-1))
            total = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent
            return total, (pg_loss, v_loss, ent)
        (tot, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, tot, aux

    pool = ctx.Pool(args.workers)
    print(f"[start] workers={args.workers} games/update={args.games_per_update} "
          f"updates={args.updates} out={args.out}", flush=True)

    rng = np.random.default_rng(args.seed + 1234)
    try:
        for upd in range(start_update, args.updates):
            sp_frac = min(1.0, (upd + 1) / max(1, args.selfplay_warmup))
            Pnp = params_to_np(state.params)
            OPnp = Pnp                                   # self-play opp = current snapshot

            # build per-game specs (seed, 3 opponent kinds)
            games = []
            for g in range(args.games_per_update):
                seed = int(rng.integers(1, 2**31 - 1))
                kinds = []
                for _ in range(3):
                    if rng.random() < sp_frac:
                        kinds.append("self")
                    else:
                        kinds.append("eff" if rng.random() < 0.7 else "rand")
                games.append((seed, kinds))
            # split across workers
            nw = args.workers
            tasks = []
            for w in range(nw):
                chunk = games[w::nw]
                if chunk:
                    tasks.append(dict(params=Pnp, opp_params=OPnp,
                                      games=chunk, shaping=args.shaping))
            t0 = time.time()
            results = pool.map(rollout_chunk, tasks)
            dt = time.time() - t0
            trajs = [tr for chunk in results for tr in chunk]

            # advantages/returns
            allO, allM, allA, allLP, allADV, allRET = [], [], [], [], [], []
            g_rews, wins, endings = [], 0, {"hu": 0, "draw": 0}
            for tr in trajs:
                adv, ret = compute_gae(tr, args.gamma, args.lam)
                allO.append(tr["obs"]); allM.append(tr["mask"]); allA.append(tr["act"])
                allLP.append(tr["logp"]); allADV.append(adv); allRET.append(ret)
                g_rews.append(float(tr["rew"].sum()))
                endings[tr["ending"]] = endings.get(tr["ending"], 0) + 1
                if tr["ending"] == "hu" and tr["winner"] == 0:
                    wins += 1
            O = np.concatenate(allO); M = np.concatenate(allM)
            A = np.concatenate(allA); LP = np.concatenate(allLP)
            ADV = np.concatenate(allADV); RET = np.concatenate(allRET)
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

            Oj = jnp.asarray(O); Mj = jnp.asarray(M); Aj = jnp.asarray(A)
            LPj = jnp.asarray(LP); ADVj = jnp.asarray(ADV); RETj = jnp.asarray(RET)
            n = O.shape[0]
            idx = np.arange(n)
            last_ent = last_pg = last_v = 0.0
            for _ in range(args.epochs):
                rng.shuffle(idx)
                for s in range(0, n, args.minibatch):
                    mb = idx[s:s + args.minibatch]
                    mbj = jnp.asarray(mb)
                    state, tot, (pg, vl, ent) = ppo_update(
                        state, Oj[mbj], Mj[mbj], Aj[mbj], LPj[mbj], ADVj[mbj], RETj[mbj])
                    last_ent, last_pg, last_v = float(ent), float(pg), float(vl)

            gps = args.games_per_update / dt
            print(f"upd {upd:4d} | R {np.mean(g_rews):+.3f} | ent {last_ent:.3f} | "
                  f"vloss {last_v:.3f} | selfwin {wins}/{len(trajs)} "
                  f"({100*wins/max(1,len(trajs)):.1f}%) | hu/draw {endings.get('hu',0)}/"
                  f"{endings.get('draw',0)} | sp_frac {sp_frac:.2f} | "
                  f"{gps:.1f} games/s | {n} transitions", flush=True)

            # --- checkpoint last every update ---
            save(last_path, state.params)
            json.dump({"update": upd + 1, "best_metric": best_metric},
                      open(meta_path, "w"))

            # --- eval ---
            if (upd + 1) % args.eval_every == 0 or upd == args.updates - 1:
                Pnp = params_to_np(state.params)
                metrics = {}
                for opp in ("rand", "eff"):
                    seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(args.eval_games)]
                    etasks = [dict(params=Pnp, opp=opp, seeds=seeds[i::args.workers])
                              for i in range(args.workers) if seeds[i::args.workers]]
                    er = [x for ch in pool.map(eval_chunk, etasks) for x in ch]
                    wr = np.mean([w for w, _, _ in er])
                    mr = np.mean([r for _, r, _ in er])
                    metrics[opp] = (wr, mr)
                print(f"  [eval upd {upd+1}] vs random: winrate {metrics['rand'][0]:.3f} "
                      f"meanR {metrics['rand'][1]:+.3f} | vs efficiency: winrate "
                      f"{metrics['eff'][0]:.3f} meanR {metrics['eff'][1]:+.3f}", flush=True)
                m = metrics["eff"][1] + metrics["rand"][1]      # combined mean reward
                if m > best_metric:
                    best_metric = m
                    save(os.path.join(args.out, "best.msgpack"), state.params)
                    json.dump({"update": upd + 1, "best_metric": best_metric},
                              open(meta_path, "w"))
                    print(f"  [ckpt] new best (metric {m:+.3f}) -> best.msgpack", flush=True)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
