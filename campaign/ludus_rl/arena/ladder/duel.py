"""Mirrored-pair match blocks (docs/04 §2): each pair = TWO games from the SAME
reset seed with sides swapped, so seat, deck, and opening-hand luck cancel in the
pair aggregate. The pair score for A is (game1 + game2) / 2 with win=1, draw=0.5.

Self-test corollary (docs/04 §1): with identical agents, game2 is bit-identical to
game1, so every pair aggregate is EXACTLY 0.5 by construction — any deviation is a
harness bug."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from boom import engine, vec
from boom.engine import TICKS_MAX


def _run_side(act_0, act_1, seeds, stream_key):
    """One batch of games: act_0 on seat 0, act_1 on seat 1, one game per seed."""
    states = vec.v_reset(jax.vmap(jax.random.PRNGKey)(seeds), None)

    def tick_fn(carry, t):
        states, key = carry
        key, k0, k1 = jax.random.split(key, 3)
        a0 = act_0(k0, states, 0, t)
        a1 = act_1(k1, states, 1, t)
        states = vec.v_step(states, jnp.stack([a0, a1], axis=1), None)
        return (states, key), None

    (states, _), _ = jax.lax.scan(tick_fn, (states, stream_key), jnp.arange(TICKS_MAX))
    return vec.v_result(states), states.illegal.sum()


@partial(jax.jit, static_argnums=(0, 1))
def _block(act_a, act_b, seeds, key):
    # SAME seeds and SAME action-RNG stream for both orders: with identical agents
    # game2 is game1 with seats relabeled, making the pair aggregate exactly 0.5
    stream = jax.random.fold_in(key, 0)
    r1, il1 = _run_side(act_a, act_b, seeds, stream)
    r2, il2 = _run_side(act_b, act_a, seeds, stream)
    s1 = jnp.where(r1 == 0, 1.0, jnp.where(r1 == 2, 0.5, 0.0))   # A was seat 0
    s2 = jnp.where(r2 == 1, 1.0, jnp.where(r2 == 2, 0.5, 0.0))   # A was seat 1
    return (s1 + s2) / 2.0, r1, r2, il1 + il2


def run_block(act_a, act_b, seeds: np.ndarray, block_key: int):
    """-> dict with per-pair aggregate scores for A and raw game results."""
    scores, r1, r2, illegal = _block(act_a, act_b, jnp.asarray(seeds, jnp.uint32),
                                     jax.random.PRNGKey(block_key))
    return {
        "pair_scores": np.asarray(scores).tolist(),
        "game1_results": np.asarray(r1).tolist(),
        "game2_results": np.asarray(r2).tolist(),
        "illegal": int(illegal),
        "seeds": np.asarray(seeds).tolist(),
    }


def self_test_exact_tie(act, n_pairs: int = 16, seed0: int = 424242) -> dict:
    """Identical agents, mirrored pairs: every aggregate must be exactly 0.5."""
    seeds = np.arange(seed0, seed0 + n_pairs, dtype=np.uint32)
    out = run_block(act, act, seeds, block_key=7)
    scores = np.asarray(out["pair_scores"])
    ok = bool((scores == 0.5).all()) and out["illegal"] == 0
    return {"pass": ok, "pairs": n_pairs,
            "deviations": int((scores != 0.5).sum()), "illegal": out["illegal"]}
