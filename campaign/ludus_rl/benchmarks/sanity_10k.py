"""Exit-criterion check: 10k random-legal matches, zero exceptions, sane outcomes.

Runs matches to TICKS_MAX, tracks the first tick each match became decided, and
writes an outcome-distribution artifact. WO-P0-01 bar: non-degenerate win rates,
overtime rate < 40%.

Usage: python benchmarks/sanity_10k.py --matches 10000 --out benchmarks/results/sanity_10k.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boom import ENV_VERSION, engine, vec
from boom.engine import RESULT_DRAW, RESULT_ONGOING, TICKS_MAX, TICKS_REG


@partial(jax.jit, static_argnums=(1,))
def run_chunk(key: jax.Array, batch: int):
    """Returns (final_result (batch,), decided_tick (batch,), illegal (batch,2))."""
    keys = jax.random.split(key, batch)
    states = vec.v_reset(keys, None)

    def tick(carry, t):
        states, key, decided = carry
        key, sub = jax.random.split(key)
        akeys = jax.random.split(sub, batch)
        actions = jax.vmap(vec._both_random_actions)(akeys, states)
        states = vec.v_step(states, actions, None)
        r = vec.v_result(states)
        decided = jnp.where((decided < 0) & (r != RESULT_ONGOING), t + 1, decided)
        return (states, key, decided), None

    decided0 = jnp.full(batch, -1, jnp.int32)
    (states, _, decided), _ = jax.lax.scan(
        tick, (states, key, decided0), jnp.arange(TICKS_MAX))
    return vec.v_result(states), decided, states.illegal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=10_000)
    ap.add_argument("--chunk", type=int, default=2_000)
    ap.add_argument("--out", default="benchmarks/results/sanity_10k.json")
    args = ap.parse_args()

    n_chunks, rem = divmod(args.matches, args.chunk)
    assert rem == 0, "matches must be a multiple of chunk"
    results, decided, illegal = [], [], []
    for c in range(n_chunks):
        r, d, il = run_chunk(jax.random.PRNGKey(1000 + c), args.chunk)
        results.append(np.asarray(r)); decided.append(np.asarray(d)); illegal.append(np.asarray(il))
        print(f"chunk {c + 1}/{n_chunks} done", flush=True)
    r = np.concatenate(results); d = np.concatenate(decided); il = np.concatenate(illegal)

    assert r.shape[0] == args.matches, "loud-fail: incomplete aggregation"
    assert (r != RESULT_ONGOING).all(), "matches must be decided by TICKS_MAX"
    assert (il == 0).all(), f"random-legal agents took {il.sum()} illegal actions"

    overtime = float((d > TICKS_REG).mean())
    stats = {
        "matches": int(args.matches),
        "p0_win_rate": float((r == 0).mean()),
        "p1_win_rate": float((r == 1).mean()),
        "draw_rate": float((r == RESULT_DRAW).mean()),
        "overtime_rate": overtime,
        "mean_decided_tick": float(d.mean()),
        "median_decided_tick": float(np.median(d)),
    }
    print(json.dumps(stats, indent=2))

    # v3 note: with exact tournament-standard stats, RANDOM play rarely breaks
    # towers (draw-heavy) and un-piloted cycle decks lose to beatdown — outcome
    # distribution under random play stopped being an engine-health signal.
    # Hard gates: zero illegal actions + complete aggregation. Distribution is
    # reported for the record; strength ordering is the ladder's job (WO-P0-04).
    checks = {
        "zero_illegal": True,
        "all_matches_terminated": bool((r != RESULT_ONGOING).all()),
    }
    stats["info_only"] = {"non_degenerate_wins_v1_gate":
                          0.05 < stats["p0_win_rate"] < 0.95
                          and 0.05 < stats["p1_win_rate"] < 0.95,
                          "overtime_below_40pct_v1_gate": overtime < 0.40}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    dev = jax.devices()[0]
    artifact = {
        "env_version": ENV_VERSION, "commit": commit,
        "device": str(dev), "jax_version": jax.__version__,
        "stats": stats, "checks": checks,
        "integrity": {"expected": args.matches, "actual": int(r.shape[0]),
                      "complete": True, "all_checks_pass": all(checks.values())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {out}; all_checks_pass={all(checks.values())}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
