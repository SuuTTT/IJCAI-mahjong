"""PPO self-play baseline for Boom (WO-P0-02), PureJaxRL-style: the whole
rollout-update loop is jit-compiled; one policy plays both seats (obs/actions are
player-centric, so weights are seat-symmetric by construction).

Reward: terminal +1/-1 (draw 0) plus a small tower-hp-differential shaping term.
Action space: flat 2305 (noop + slot*tile), invalid actions masked to -inf.

Run (on a GPU box):
    python baselines/ppo_selfplay.py --updates 20000 --out /root/ludus_train/ppo_v0
Artifacts: params_<update>.msgpack + log.jsonl (one JSON line per log/eval event,
with commit + config embedded — AGENTS.md provenance).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import to_bytes

from boom import engine, vec
from boom.cards import DECK_A, DECK_B
from boom.engine import H, OBS_C, OBS_VEC, RESULT_ONGOING, W

N_ACTIONS = vec.N_ACTIONS
# per-side tower hp pool, derived from the engine so balance patches can't desync it
TOWER_HP_TOTAL = float(engine.TOWER_MAX_HP[:3].sum())


# ---------------------------------------------------------------- model
class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, spatial, vector):
        x = nn.relu(nn.Conv(24, (3, 3), strides=2)(spatial))
        x = nn.relu(nn.Conv(48, (3, 3), strides=2)(x))
        x = x.reshape((*x.shape[:-3], -1))
        x = nn.relu(nn.Dense(256)(x))
        v = nn.relu(nn.Dense(128)(vector))
        h = nn.relu(nn.Dense(256)(jnp.concatenate([x, v], axis=-1)))
        logits = nn.Dense(N_ACTIONS, kernel_init=nn.initializers.orthogonal(0.01))(h)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(h)
        return logits, value.squeeze(-1)


class Transition(NamedTuple):
    obs_s: jax.Array      # (2N, H, W, C)
    obs_v: jax.Array      # (2N, OBS_VEC)
    mask: jax.Array       # (2N, A) bool
    action: jax.Array     # (2N,)
    logp: jax.Array       # (2N,)
    value: jax.Array      # (2N,)
    reward: jax.Array     # (2N,)
    done: jax.Array       # (2N,)


def tower_score(states, player):
    """Own-minus-enemy tower hp, normalized to [-1, 1], per env."""
    own = jnp.sum(jnp.where(jnp.asarray(engine.TOWER_OWNER)[None, :] == player,
                            states.tower_hp, 0), axis=1)
    opp = jnp.sum(jnp.where(jnp.asarray(engine.TOWER_OWNER)[None, :] != player,
                            states.tower_hp, 0), axis=1)
    return (own - opp) / TOWER_HP_TOTAL


def both_obs(states):
    o0 = vec.v_observe(states, 0)
    o1 = vec.v_observe(states, 1)
    spatial = jnp.concatenate([o0.spatial, o1.spatial])
    vector = jnp.concatenate([o0.vector, o1.vector])
    mask = jnp.concatenate([jax.vmap(vec.flat_legal, in_axes=(0, None))(states, 0),
                            jax.vmap(vec.flat_legal, in_axes=(0, None))(states, 1)])
    return spatial, vector, mask


DECK_OPTIONS = None  # built lazily to avoid jax init at import


def deck_pair(cfg):
    import jax.numpy as jnp
    if cfg.decks in ("mixed", "mirror"):
        return None                      # per-env randomized in make_train
    d = {"A": DECK_A, "B": DECK_B}
    return jnp.stack([jnp.asarray(d[cfg.decks[0]]), jnp.asarray(d[cfg.decks[1]])])


def make_train(cfg):
    net = ActorCritic()
    decks = deck_pair(cfg)
    mixed = cfg.decks == "mixed"
    mirror = cfg.decks == "mirror"
    options = jnp.stack([jnp.asarray(DECK_A), jnp.asarray(DECK_B)])
    v_reset_per = jax.vmap(engine.reset, in_axes=(0, 0))

    def fresh_states(reset_keys):
        """mixed mode: every env deals each seat a random deck — the policy must
        learn to pilot ALL decks (fixes the deck-specialist failure of ppo_v0)."""
        if mirror:
            # both seats dealt the SAME random deck -> balanced mirror matchups
            dk = jax.vmap(lambda k: jax.random.randint(k, (1,), 0, 2))(reset_keys)
            dk2 = jnp.concatenate([dk, dk], axis=1)      # (N,2) same deck both seats
            return v_reset_per(reset_keys, options[dk2])
        if mixed:
            dk = jax.vmap(lambda k: jax.random.randint(k, (2,), 0, 2))(reset_keys)
            return v_reset_per(reset_keys, options[dk])
        return vec.v_reset(reset_keys, decks)

    def act(params, spatial, vector, mask, key):
        logits, value = net.apply(params, spatial, vector)
        logits = jnp.where(mask, logits, -1e9)
        action = jax.random.categorical(key, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        return action, logp, value

    def env_step(carry, _):
        # carry: (states, key, params, opp_params, use_pool, step_i)
        # mirror mode: seat 1 shares the learner's params (classic self-play).
        # pool mode:   seat 1 plays a FROZEN opponent (a sampled league gen) —
        # breaks the mirror equilibrium that stalled the league at gen 4.
        states, key, params, opp_params, use_pool, step_i = carry
        N = cfg.num_envs
        spatial, vector, mask = both_obs(states)
        key, ka, ko = jax.random.split(key, 3)
        action, logp, value = act(params, spatial, vector, mask, ka)
        opp_action, _, _ = act(opp_params, spatial[N:], vector[N:], mask[N:], ko)
        action = jnp.where(use_pool,
                           jnp.concatenate([action[:N], opp_action]), action)

        a0 = jax.vmap(vec.flat_to_triple)(action[:N])
        a1 = jax.vmap(vec.flat_to_triple)(action[N:])
        actions = jnp.stack([a0, a1], axis=1)          # (N, 2, 3)

        score0_pre = tower_score(states, 0)
        new_states = vec.v_step(states, actions, None)
        res = vec.v_result(new_states)
        done = res != RESULT_ONGOING
        score0_post = tower_score(new_states, 0)

        shaping0 = (score0_post - score0_pre) * cfg.shaping
        term0 = jnp.where(res == 0, 1.0, jnp.where(res == 1, -1.0, 0.0))
        r0 = jnp.where(done, term0, shaping0)
        # elixir-overflow penalty: a tick spent capped at max elixir wastes regen
        # (objectively bad in CR). Without it, self-play converged to a passive
        # equilibrium — after one tower fell, BOTH sides hoarded to game end.
        # Applied per seat (not zero-sum): passivity costs everyone.
        overflow = (states.energy >= engine.E_MAX).astype(jnp.float32)  # (N, 2)
        pen = cfg.overflow_pen * jnp.where(done[:, None], 0.0, overflow)
        reward = jnp.concatenate([r0, -r0]) - jnp.concatenate([pen[:, 0], pen[:, 1]])

        # auto-reset finished envs (fresh keys derived from a running counter)
        key, kr = jax.random.split(key)
        reset_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
            kr, step_i * N + jnp.arange(N))
        fresh = fresh_states(reset_keys)
        states = jax.tree_util.tree_map(
            lambda new, fr: jnp.where(
                done.reshape((-1,) + (1,) * (new.ndim - 1)), fr, new),
            new_states, fresh)

        t = Transition(spatial, vector, mask, action, logp, value,
                       reward, jnp.concatenate([done, done]).astype(jnp.float32))
        return (states, key, params, opp_params, use_pool, step_i + 1), t

    def compute_gae(traj, last_value):
        def scan_fn(carry, t):
            gae, next_value = carry
            delta = t.reward + cfg.gamma * next_value * (1 - t.done) - t.value
            gae = delta + cfg.gamma * cfg.lam * (1 - t.done) * gae
            return (gae, t.value), gae
        (_, _), advs = jax.lax.scan(scan_fn, (jnp.zeros_like(last_value), last_value),
                                    traj, reverse=True)
        return advs, advs + traj.value

    def loss_fn(params, batch, adv, target, valid):
        logits, value = net.apply(params, batch.obs_s, batch.obs_v)
        logits = jnp.where(batch.mask, logits, -1e9)
        logp_all = jax.nn.log_softmax(logits)
        logp = logp_all[jnp.arange(batch.action.shape[0]), batch.action]
        ratio = jnp.exp(logp - batch.logp)
        w = valid / jnp.maximum(valid.sum(), 1.0)
        adv_n = (adv - (adv * w).sum()) / (jnp.sqrt(((adv - (adv * w).sum()) ** 2 * w).sum()) + 1e-8)
        pg = -(jnp.minimum(ratio * adv_n,
                           jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_n) * w).sum()
        v_loss = 0.5 * (jnp.square(value - target) * w).sum()
        probs = jnp.exp(logp_all)
        ent = ((-jnp.where(batch.mask, probs * logp_all, 0.0).sum(-1)) * w).sum()
        return pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent, (pg, v_loss, ent)

    @jax.jit   # no donation: mirror mode passes runner's own params as opp_params
    def train_update(runner, opp_params, use_pool):
        params, opt_state, states, key, step_i = runner
        (states, key, _, _, _, step_i), traj = jax.lax.scan(
            env_step, (states, key, params, opp_params, use_pool, step_i),
            None, cfg.rollout_t)
        spatial, vector, mask = both_obs(states)
        _, last_value = net.apply(params, spatial, vector)
        adv, target = compute_gae(traj, last_value)

        B = cfg.rollout_t * 2 * cfg.num_envs
        flat = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), traj)
        adv, target = adv.reshape(B), target.reshape(B)
        # pool mode: seat-1 rows carry the frozen opponent's actions — exclude
        seat1 = (jnp.arange(cfg.rollout_t * 2 * cfg.num_envs)
                 // cfg.num_envs) % 2 == 1
        valid_all = jnp.where(use_pool & seat1, 0.0, 1.0)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, B)
            def minibatch(carry, idx):
                params, opt_state = carry
                mb = jax.tree_util.tree_map(lambda x: x[idx], flat)
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, mb, adv[idx], target[idx], valid_all[idx])
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), (loss, *aux)
            idxs = perm.reshape(cfg.minibatches, -1)
            (params, opt_state), metrics = jax.lax.scan(
                minibatch, (params, opt_state), idxs)
            return (params, opt_state, key), metrics
        (params, opt_state, key), metrics = jax.lax.scan(
            epoch, (params, opt_state, key), None, cfg.epochs)

        ep_done = traj.done.sum()
        mean_r = traj.reward.sum() / jnp.maximum(ep_done, 1)
        stats = {"loss": metrics[0].mean(), "pg": metrics[1].mean(),
                 "v": metrics[2].mean(), "ent": metrics[3].mean(),
                 "episodes": ep_done / 2, "terminal_r_per_ep": mean_r}
        return (params, opt_state, states, key, step_i), stats

    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr, eps=1e-5))
    return net, tx, train_update


# ---------------------------------------------------------------- evaluation
def make_eval(net, cfg):
    decks = deck_pair(cfg)   # mixed -> None = default A-vs-B: measures generalization

    @partial(jax.jit, static_argnums=(2,))
    def eval_vs_random(params, key, n_matches):
        keys = jax.random.split(key, n_matches)
        states = vec.v_reset(keys, decks)

        def tick(carry, _):
            states, key = carry
            o0 = vec.v_observe(states, 0)
            m0 = jax.vmap(vec.flat_legal, in_axes=(0, None))(states, 0)
            logits, _ = net.apply(params, o0.spatial, o0.vector)
            a0f = jnp.argmax(jnp.where(m0, logits, -1e9), axis=-1)   # greedy
            key, kb = jax.random.split(key)
            bkeys = jax.random.split(kb, n_matches)
            a1f = jax.vmap(lambda k, s: vec.random_legal_action(k, s, 1))(bkeys, states)
            a0 = jax.vmap(vec.flat_to_triple)(a0f)
            a1 = jax.vmap(vec.flat_to_triple)(a1f)
            states = vec.v_step(states, jnp.stack([a0, a1], axis=1), None)
            return (states, key), None

        (states, _), _ = jax.lax.scan(tick, (states, key), None, engine.TICKS_MAX)
        res = vec.v_result(states)
        return (res == 0).mean(), (res == 1).mean(), (res == 2).mean()
    return eval_vs_random


def wilson_lower(p, n, z=1.96):
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (center - rad) / denom


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--rollout-t", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=4e-2)
    ap.add_argument("--decks", choices=["BB", "AA", "AB", "mixed", "mirror"], default="mixed",
                    help="training/eval deck pairing; symmetric mirrors (BB/AA) avoid "
                         "the asymmetric-deck curriculum problem on v3 stats")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--shaping", type=float, default=1.0)
    ap.add_argument("--overflow-pen", type=float, default=0.002,
                    help="per-tick penalty for idling at max elixir (kills the "
                         "passive endgame equilibrium)")
    ap.add_argument("--pool-dir", default=None,
                    help="dir of frozen opponent checkpoints (league gens); "
                         "when set, half the updates train vs a sampled one")
    ap.add_argument("--pool-frac", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-matches", type=int, default=256)
    ap.add_argument("--out", default="/root/ludus_train/ppo_v0")
    ap.add_argument("--resume", default=None,
                    help="msgpack checkpoint to initialize parameters from")
    cfg = ap.parse_args()

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    log_f = open(out / "log.jsonl", "a")

    def log(obj):
        obj["t"] = time.time()
        log_f.write(json.dumps(obj) + "\n")
        log_f.flush()
        print(json.dumps(obj), flush=True)

    log({"event": "start", "commit": commit, "config": vars(cfg),
         "device": str(jax.devices()[0])})

    net, tx, train_update = make_train(cfg)
    eval_fn = make_eval(net, cfg)

    key = jax.random.PRNGKey(0)
    key, ki, ke = jax.random.split(key, 3)
    dummy_s = jnp.zeros((1, H, W, OBS_C), jnp.float32)
    dummy_v = jnp.zeros((1, OBS_VEC), jnp.float32)
    params = net.init(ki, dummy_s, dummy_v)
    if cfg.resume:
        from flax.serialization import from_bytes
        params = from_bytes(params, Path(cfg.resume).read_bytes())
        log({"event": "resumed_from", "path": cfg.resume})
    opt_state = tx.init(params)
    if cfg.decks == "mixed":
        rk = jax.random.split(ke, cfg.num_envs)
        opts = jnp.stack([jnp.asarray(DECK_A), jnp.asarray(DECK_B)])
        dk = jax.vmap(lambda k: jax.random.randint(k, (2,), 0, 2))(rk)
        states = jax.vmap(engine.reset, in_axes=(0, 0))(rk, opts[dk])
    else:
        states = vec.v_reset(jax.random.split(ke, cfg.num_envs), deck_pair(cfg))
    runner = (params, opt_state, states, key, jnp.int32(1))

    # opponent pool: frozen league generations (PFSP-lite)
    pool = []
    if cfg.pool_dir:
        from flax.serialization import from_bytes
        for p in sorted(Path(cfg.pool_dir).glob("gen_*.msgpack")):
            try:
                pool.append(from_bytes(params, p.read_bytes()))
            except Exception:
                pass
        log({"event": "pool_loaded", "size": len(pool)})
    import random as _random
    _random.seed(0)

    steps_per_update = cfg.num_envs * cfg.rollout_t
    t0 = time.time()
    for u in range(1, cfg.updates + 1):
        use_pool = bool(pool) and _random.random() < cfg.pool_frac
        opp = _random.choice(pool) if use_pool else runner[0]
        runner, stats = train_update(runner, opp, jnp.bool_(use_pool))
        if u % cfg.log_every == 0:
            stats = {k: float(v) for k, v in stats.items()}
            sps = steps_per_update * cfg.log_every / max(time.time() - t0, 1e-9)
            t0 = time.time()
            log({"event": "train", "update": u,
                 "env_steps": u * steps_per_update, "env_steps_per_s": sps, **stats})
        if u % cfg.eval_every == 0 or u == cfg.updates:
            params = runner[0]
            w, l, d = eval_fn(params, jax.random.PRNGKey(u), cfg.eval_matches)
            w, l, d = float(w), float(l), float(d)
            log({"event": "eval", "update": u, "vs": "random_v0",
                 "matches": cfg.eval_matches, "win": w, "loss": l, "draw": d,
                 "win_ci_lower": wilson_lower(w, cfg.eval_matches)})
            (out / f"params_{u}.msgpack").write_bytes(to_bytes(params))
            (out / "params_latest.msgpack").write_bytes(to_bytes(params))
    log({"event": "done", "updates": cfg.updates})


if __name__ == "__main__":
    main()
