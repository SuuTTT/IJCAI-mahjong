"""Vmapped batch API + flat action encoding + random-legal rollouts.

Flat action id (per player): 0 = noop; 1 + (slot * W*H + x * H + y) for slot in 0..3.
Total: 1 + 4*18*32 = 2305 discrete actions, always player-centric.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import engine
from .engine import H, HAND, W, State

N_ACTIONS = 1 + HAND * W * H


def flat_to_triple(a: jax.Array) -> jax.Array:
    """Flat action id -> [slot, x, y] (noop -> slot 4)."""
    i = a - 1
    slot = jnp.where(a == 0, HAND, i // (W * H))
    x = jnp.where(a == 0, 0, (i // H) % W)
    y = jnp.where(a == 0, 0, i % H)
    return jnp.stack([slot, x, y]).astype(jnp.int32)


def flat_legal(state: State, player) -> jax.Array:
    """(N_ACTIONS,) bool mask matching flat_to_triple."""
    m = engine.legal(state, player)                     # (5, W, H)
    return jnp.concatenate([jnp.ones(1, bool), m[:HAND].reshape(-1)])


def random_legal_action(key: jax.Array, state: State, player) -> jax.Array:
    """Uniform sample over the flat legal mask (noop is one action among many,
    so the random agent plays cards aggressively — good for exercising mechanics)."""
    mask = flat_legal(state, player)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits)


# vmapped core (leading batch axis on state / keys / actions)
v_reset = jax.vmap(engine.reset, in_axes=(0, None))
v_step = jax.vmap(engine.step, in_axes=(0, 0, None))
v_result = jax.vmap(engine.result)
v_observe = jax.vmap(engine.observe, in_axes=(0, None))
v_legal = jax.vmap(engine.legal, in_axes=(0, None))


def _both_random_actions(key: jax.Array, state: State) -> jax.Array:
    k0, k1 = jax.random.split(key)
    a0 = flat_to_triple(random_legal_action(k0, state, 0))
    a1 = flat_to_triple(random_legal_action(k1, state, 1))
    return jnp.stack([a0, a1])


def rollout_random(key: jax.Array, batch: int, ticks: int,
                   decks: jax.Array | None = None,
                   record: bool = False):
    """Run `batch` matches for `ticks` steps with both players random-legal.

    Returns (final_states, action_log) — action_log is (ticks, batch, 2, 3) if
    record else None. Fully jit-compiled including the scan."""
    keys = jax.random.split(key, batch)
    states = v_reset(keys, decks)

    def tick(carry, t):
        states, key = carry
        key, sub = jax.random.split(key)
        akeys = jax.random.split(sub, batch)
        actions = jax.vmap(_both_random_actions)(akeys, states)
        states = v_step(states, actions, None)
        return (states, key), (actions if record else None)

    (states, _), log = jax.lax.scan(tick, (states, key), jnp.arange(ticks))
    return states, log


rollout_random_jit = jax.jit(rollout_random, static_argnums=(1, 2, 4))


def replay(seed_key: jax.Array, action_log: jax.Array,
           decks: jax.Array | None = None) -> State:
    """Deterministic replay of a single match from (key, action_log (T, 2, 3))."""
    state = engine.reset(seed_key, decks)

    def tick(state, actions):
        return engine.step(state, actions, None), None

    state, _ = jax.lax.scan(tick, state, action_log)
    return state


replay_jit = jax.jit(replay, static_argnums=())
