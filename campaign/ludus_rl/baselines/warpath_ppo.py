"""Self-play PPO for Warpath over the macro action layer (warpath/rl.py).

Compact vector-MLP actor-critic. Both seats act with the current policy
(symmetric self-play); we collect experience from BOTH seats each step, so one
rollout yields 2*T*B transitions. Frame-skip holds each macro for K engine ticks
to coarsen the decision rate and lengthen the effective horizon.

Eval each checkpoint against two stationary references: the scripted rule_v0
commander and a uniform-random macro policy.

Run:
  python -m baselines.warpath_ppo --updates 3000 --num-envs 256 --rollout-t 48 \
      --frameskip 8 --eval-every 50 --out /root/ludus_train/warpath/agent
"""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import flax.linen as nn
import flax.serialization as fser
import jax
import jax.numpy as jnp
import numpy as np
import optax

from warpath import engine as E, rl

OBS, NACT = rl.OBS_DIM, rl.N_ACT


class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(256)(x))
        h = nn.relu(nn.Dense(256)(h))
        logits = nn.Dense(NACT, kernel_init=nn.initializers.orthogonal(0.01))(h)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(h)
        return logits, value[..., 0]


# ---- vectorised engine + harness --------------------------------------
v_reset = jax.vmap(lambda k: E.reset())
v_step = jax.vmap(E.step)
v_res = jax.vmap(E.result)
v_obs = lambda st, p: jax.vmap(lambda s: rl.obs(s, p))(st)
v_act = lambda st, p, m: jax.vmap(lambda s, mm: rl.macro_action(s, p, mm))(st, m)
v_scr = lambda st, p: jax.vmap(lambda s: rl.scripted_macro(s, p))(st)
v_rew = lambda a, b, p: jax.vmap(lambda x, y: rl.reward(x, y, p))(a, b)


def _reset_done(st, done, keys):
    """Replace finished envs with fresh ones (auto-reset)."""
    fresh = v_reset(keys)
    pick = lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)
    return jax.tree.map(pick, fresh, st)


def make_step_fns(cfg):
    net = ActorCritic()

    def policy(params, obs, key):
        logits, val = net.apply(params, obs)
        act = jax.random.categorical(key, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(act.shape[0]), act]
        return act, logp, val

    def env_advance(st, m0, m1, key):
        """Hold both macros for `frameskip` engine ticks; sum reward, auto-reset."""
        def one(carry, _):
            st, rew0, rew1, dead = carry
            a0, a1 = v_act(st, 0, m0), v_act(st, 1, m1)
            nxt = v_step(st, jnp.stack([a0, a1], axis=1))
            live = ~dead
            rew0 = rew0 + jnp.where(live, v_rew(st, nxt, 0), 0.0)
            rew1 = rew1 + jnp.where(live, v_rew(st, nxt, 1), 0.0)
            dead = dead | (v_res(nxt) != -1)
            return (nxt, rew0, rew1, dead), None
        z = jnp.zeros(cfg.num_envs)
        (st, r0, r1, dead), _ = jax.lax.scan(
            one, (st, z, z, jnp.zeros(cfg.num_envs, bool)), None, length=cfg.frameskip)
        done = v_res(st) != -1
        keys = jax.random.split(key, cfg.num_envs)
        st = _reset_done(st, done, keys)
        return st, r0, r1, done

    return net, policy, env_advance


def rollout(cfg, net, policy, env_advance, params, st, key, opp_params):
    """Agent controls p0. Opponent (p1) is: the scripted commander ('scripted'),
    the current policy ('self'), or a FROZEN pool member whose params are passed
    in `opp_params` ('pool'). Experience collected from p0 only."""
    def step(carry, _):
        st, key = carry
        key, k0, k1, ke = jax.random.split(key, 4)
        o0 = v_obs(st, 0)
        a0, lp0, v0 = policy(params, o0, k0)
        if cfg.opponent == "self":
            a1, _, _ = policy(params, v_obs(st, 1), k1)
        elif cfg.opponent == "pool":
            logits1, _ = net.apply(opp_params, v_obs(st, 1))   # frozen past self
            a1 = jax.random.categorical(k1, logits1)
        else:
            a1 = v_scr(st, 1)
        nst, r0, r1, done = env_advance(st, a0, a1, ke)
        tr = dict(obs=o0, act=a0, logp=lp0, val=v0, rew=r0, done=done)
        return (nst, key), tr
    (st, key), traj = jax.lax.scan(step, (st, key), None, length=cfg.rollout_t)
    lastv = net.apply(params, v_obs(st, 0))[1]           # bootstrap value
    return st, key, traj, lastv


def gae(traj, lastv, gamma, lam):
    def scan(carry, x):
        gae, nextv = carry
        rew, val, done = x
        delta = rew + gamma * nextv * (1 - done) - val
        gae = delta + gamma * lam * (1 - done) * gae
        return (gae, val), gae
    rew, val, done = traj["rew"], traj["val"], traj["done"].astype(jnp.float32)
    _, adv = jax.lax.scan(scan, (jnp.zeros_like(lastv), lastv),
                          (rew, val, done), reverse=True)
    return adv, adv + val


def make_update(cfg, net, policy, env_advance, opt):
    def loss_fn(params, batch, ent_c):
        logits, val = net.apply(params, batch["obs"])
        logp_all = jax.nn.log_softmax(logits)
        logp = logp_all[jnp.arange(batch["act"].shape[0]), batch["act"]]
        ratio = jnp.exp(logp - batch["logp"])
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv,
                          jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vloss = 0.5 * ((val - batch["ret"]) ** 2).mean()
        ent = -(logp_all * jnp.exp(logp_all)).sum(-1).mean()
        return pg + cfg.vf * vloss - ent_c * ent, (pg, vloss, ent)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def update(params, opt_state, st, key, ent_c, opp_params):
        st, key, traj, lastv = rollout(cfg, net, policy, env_advance, params, st, key,
                                       opp_params)
        adv, ret = gae(traj, lastv, cfg.gamma, cfg.lam)
        flat = lambda a: a.reshape((-1,) + a.shape[3:]) if a.ndim > 3 else a.reshape(-1)
        data = dict(obs=traj["obs"].reshape(-1, OBS), act=flat(traj["act"]),
                    logp=flat(traj["logp"]), adv=flat(adv), ret=flat(ret))
        B = data["act"].shape[0]

        def epoch(carry, _):
            params, opt_state, key = carry
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, B).reshape(cfg.minibatches, -1)

            def mb(carry, idx):
                params, opt_state = carry
                batch = jax.tree.map(lambda a: a[idx], data)
                (l, aux), g = grad_fn(params, batch, ent_c)
                updates, opt_state = opt.update(g, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), aux
            (params, opt_state), aux = jax.lax.scan(mb, (params, opt_state), perm)
            return (params, opt_state, key), aux
        (params, opt_state, key), aux = jax.lax.scan(
            epoch, (params, opt_state, key), None, length=cfg.epochs)
        return params, opt_state, st, key, traj["rew"].mean(), aux[2].mean()
    return update


def eval_vs(cfg, net, params, opp, key, n=256, ticks=3000):
    """Agent = p0; opponent = 'scripted' or 'random'. Both play SAMPLED (temp 1)
    so — despite the deterministic start — the n games diverge into a real
    win-rate distribution instead of one replicated deterministic match."""
    st = v_reset(jax.random.split(key, n))
    def body(carry, _):
        st, key = carry
        key, ka, kr = jax.random.split(key, 3)
        logits, _ = net.apply(params, jax.vmap(lambda s: rl.obs(s, 0))(st))
        m0 = jax.random.categorical(ka, logits)
        if opp == "scripted":
            m1 = jax.vmap(lambda s: rl.scripted_macro(s, 1))(st)
        else:
            m1 = jax.random.randint(kr, (n,), 0, NACT)
        a0 = jax.vmap(lambda s, m: rl.macro_action(s, 0, m))(st, m0)
        a1 = jax.vmap(lambda s, m: rl.macro_action(s, 1, m))(st, m1)
        st = jax.vmap(E.step)(st, jnp.stack([a0, a1], axis=1))
        return (st, key), None
    (st, _), _ = jax.lax.scan(body, (st, key), None, length=ticks)
    res = jax.vmap(E.result)(st)
    return jnp.mean(res == 0)


def eval_vs_pool(net, params, opp_params, key, n=256, ticks=3000):
    """Transitivity: agent (p0) vs a FROZEN pool member (p1), both SAMPLED.
    Returns p0's win rate. >0.5 => the current agent is stronger than that
    ancestor — the non-saturating progress signal a fixed baseline can't give."""
    st = v_reset(jax.random.split(key, n))
    def body(carry, _):
        st, key = carry
        key, ka, kb = jax.random.split(key, 3)
        l0, _ = net.apply(params, jax.vmap(lambda s: rl.obs(s, 0))(st))
        l1, _ = net.apply(opp_params, jax.vmap(lambda s: rl.obs(s, 1))(st))
        m0 = jax.random.categorical(ka, l0)
        m1 = jax.random.categorical(kb, l1)
        a0 = jax.vmap(lambda s, m: rl.macro_action(s, 0, m))(st, m0)
        a1 = jax.vmap(lambda s, m: rl.macro_action(s, 1, m))(st, m1)
        st = jax.vmap(E.step)(st, jnp.stack([a0, a1], axis=1))
        return (st, key), None
    (st, _), _ = jax.lax.scan(body, (st, key), None, length=ticks)
    res = jax.vmap(E.result)(st)
    return jnp.mean(res == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--rollout-t", type=int, default=48)
    ap.add_argument("--frameskip", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)   # horizon > game length
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)       # annealed to --ent-final
    ap.add_argument("--ent-final", type=float, default=0.001)
    ap.add_argument("--opponent", choices=["scripted", "self", "pool"],
                    default="scripted",
                    help="p1 during training: scripted = rule_v0 curriculum, "
                         "self = naive self-play, pool = league vs frozen past selves")
    ap.add_argument("--snapshot-every", type=int, default=200,
                    help="pool mode: snapshot the agent into the pool every N updates")
    ap.add_argument("--pool-max", type=int, default=12, help="max pool members kept")
    ap.add_argument("--pool-dir", default="/root/ludus_train/warpath/pool")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--out", default="/root/ludus_train/warpath/agent")
    ap.add_argument("--resume", default=None)
    cfg = ap.parse_args()
    import random as pyrandom

    key = jax.random.PRNGKey(0)
    key, ki = jax.random.split(key)
    net, policy, env_advance = make_step_fns(cfg)
    params = net.init(ki, jnp.zeros((1, OBS)))
    if cfg.resume and Path(cfg.resume).exists():
        params = fser.from_bytes(params, Path(cfg.resume).read_bytes())
        print(f"resumed from {cfg.resume}", flush=True)
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr))
    opt_state = opt.init(params)
    update = make_update(cfg, net, policy, env_advance, opt)

    st = v_reset(jax.random.split(key, cfg.num_envs))
    out = Path(cfg.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best = -1.0
    bp = Path(cfg.out + ".best.txt")           # resume best so restarts don't clobber
    if bp.exists():
        best = float(bp.read_text().strip() or -1.0)

    # ---- opponent pool (league) ----
    pool = []
    pool_dir = Path(cfg.pool_dir)
    if cfg.opponent == "pool":
        pool_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(pool_dir.glob("pool_*.msgpack")):
            pool.append(fser.from_bytes(params, f.read_bytes()))
        if not pool:                            # seed with the (resumed) agent
            pool.append(jax.tree.map(lambda x: x, params))
            (pool_dir / "pool_000000.msgpack").write_bytes(fser.to_bytes(params))
        print(f"pool: {len(pool)} members", flush=True)

    def sample_opp():
        if cfg.opponent == "pool":
            return pool[pyrandom.randrange(len(pool))]     # uniform fictitious self-play
        return params                                      # unused for scripted/self

    for u in range(1, cfg.updates + 1):
        frac = u / cfg.updates
        ent_c = jnp.float32(cfg.ent * (1 - frac) + cfg.ent_final * frac)  # anneal
        params, opt_state, st, key, mrew, ent = update(
            params, opt_state, st, key, ent_c, sample_opp())
        if u % 10 == 0:
            mrew.block_until_ready()
            sps = u * cfg.rollout_t * cfg.num_envs * cfg.frameskip * 2 / (time.time() - t0)
            print(f"u{u:>4} rew {float(mrew):+.3f} ent {float(ent):.3f} "
                  f"entc {float(ent_c):.4f} {sps/1e6:.2f}M steps/s", flush=True)
        # snapshot into the pool
        if cfg.opponent == "pool" and u % cfg.snapshot_every == 0:
            pool.append(jax.tree.map(lambda x: x, params))
            (pool_dir / f"pool_{u:06d}.msgpack").write_bytes(fser.to_bytes(params))
            if len(pool) > cfg.pool_max:        # keep the seed ancestor + most recent
                pool[:] = [pool[0]] + pool[-(cfg.pool_max - 1):]
            print(f"  [snapshot u{u}] pool now {len(pool)} members", flush=True)
        if u % cfg.eval_every == 0:
            key, k1, k2, k3, k4 = jax.random.split(key, 5)
            ws = float(eval_vs(cfg, net, params, "scripted", k1))
            wr = float(eval_vs(cfg, net, params, "random", k2))
            line = f"  [eval u{u}] win_vs_scripted {ws:.3f}  win_vs_random {wr:.3f}"
            if cfg.opponent == "pool" and len(pool) > 1:
                w_old = float(eval_vs_pool(net, params, pool[0], k3))    # vs oldest
                w_new = float(eval_vs_pool(net, params, pool[-1], k4))   # vs newest
                line += f"  vs_ancestor {w_old:.3f}  vs_recent {w_new:.3f}"
            score = ws + wr
            Path(cfg.out + ".last.msgpack").write_bytes(fser.to_bytes(params))
            if score > best:
                best = score
                Path(cfg.out + ".msgpack").write_bytes(fser.to_bytes(params))
                bp.write_text(str(best))
                line += "  <- new best (saved)"
            print(line, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
