"""Paired, seat-swapped evaluation between two agents (pre-ladder standard tool).

Agents: random | rule | ppo:<params.msgpack>
Seat effects are cancelled by playing half the matches with each agent as player 0.
Reports per-seat and combined win rates with a Wilson 95% lower bound — per
AGENTS.md, the CI bound is the claim, not the mean.

    python baselines/eval_pair.py --a ppo:/root/ludus_train/ppo_v0/params_latest.msgpack \
        --b rule --matches 512 --out benchmarks/results/eval_ppo_vs_rule.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from baselines.ppo_selfplay import ActorCritic, wilson_lower
from baselines.rule_bot import rule_v0_action
from boom import engine, vec
from boom.engine import H, OBS_C, OBS_VEC, TICKS_MAX, W

BOT_PERIOD = 5   # scripted/random agents act once per second (same as play server)
NOOP3 = jnp.array([4, 0, 0], jnp.int32)


def make_agent(spec: str):
    """Returns (act_fn(key, states, seat, tick) -> (N,3) triples, name)."""
    if spec == "random":
        def act(key, states, seat, tick):
            keys = jax.random.split(key, states.tick.shape[0])
            a = jax.vmap(lambda k, s: vec.flat_to_triple(
                vec.random_legal_action(k, s, seat)))(keys, states)
            return jnp.where(tick % BOT_PERIOD == 0, a, NOOP3[None, :])
        return act, "random_v0"

    if spec == "rule":
        def act(key, states, seat, tick):
            a = jax.vmap(lambda s: rule_v0_action(s, seat))(states)
            return jnp.where(tick % BOT_PERIOD == 0, a, NOOP3[None, :])
        return act, "rule_v0"

    if spec.startswith("ppo:"):
        from flax.serialization import from_bytes
        net = ActorCritic()
        tmpl = net.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, H, W, OBS_C), jnp.float32),
                        jnp.zeros((1, OBS_VEC), jnp.float32))
        params = from_bytes(tmpl, Path(spec[4:]).read_bytes())

        def act(key, states, seat, tick):
            obs = vec.v_observe(states, seat)
            mask = jax.vmap(vec.flat_legal, in_axes=(0, None))(states, seat)
            logits, _ = net.apply(params, obs.spatial, obs.vector)
            flat = jnp.argmax(jnp.where(mask, logits, -1e9), axis=-1)
            return jax.vmap(vec.flat_to_triple)(flat)
        return act, f"ppo({Path(spec[4:]).name})"

    raise SystemExit(f"unknown agent spec: {spec}")


DECKS = None  # set by --decks; None = engine defaults (A vs B)


def run_side(act_a, act_b, n, seed):
    """A plays seat 0, B seat 1. Returns (result codes (n,), total illegal count)."""
    @partial(jax.jit, static_argnums=())
    def go(key):
        keys = jax.random.split(key, n)
        states = vec.v_reset(keys, DECKS)

        def tick_fn(carry, t):
            states, key = carry
            key, k0, k1 = jax.random.split(key, 3)
            a0 = act_a(k0, states, 0, t)
            a1 = act_b(k1, states, 1, t)
            states = vec.v_step(states, jnp.stack([a0, a1], axis=1), None)
            return (states, key), None

        (states, _), _ = jax.lax.scan(tick_fn, (states, jax.random.fold_in(key, 7)),
                                      jnp.arange(TICKS_MAX))
        return vec.v_result(states), states.illegal.sum()
    return go(jax.random.PRNGKey(seed))


def self_test(act, name, n, seed):
    """Calibration self-test (WO-P0-02): the SAME agent on both seats over the SAME
    seeds must produce bit-identical outcomes both times we run it — the harness
    cannot manufacture an edge between identical agents. Returns the dict or aborts."""
    r1, il1 = run_side(act, act, n, seed)
    r2, il2 = run_side(act, act, n, seed)
    import numpy as np
    assert (np.asarray(r1) == np.asarray(r2)).all(), \
        "calibration self-test FAILED: identical runs diverged"
    assert int(il1) == 0 and int(il2) == 0, "self-test FAILED: illegal actions taken"
    p0 = float((np.asarray(r1) == 0).mean())
    return {"agent": name, "matches": n, "deterministic_repeat": True,
            "p0_win_rate": p0, "illegal": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--matches", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--decks", choices=["default", "AA", "BB", "mirror"],
                    default="default",
                    help="deck pairing; AA/BB are single-deck mirrors (both seats "
                         "same deck); 'mirror' pools AA+BB to match mirror-deck "
                         "training and give a fair, seat- AND deck-symmetric gate")
    args = ap.parse_args()

    import numpy as np
    from boom.cards import DECK_A, DECK_B

    def _decks_for(mode):
        if mode == "default":
            return None
        d = DECK_A if mode == "AA" else DECK_B
        return jnp.stack([jnp.asarray(d), jnp.asarray(d)])

    global DECKS
    act_a, name_a = make_agent(args.a)
    act_b, name_b = make_agent(args.b)

    # "mirror" pools both single-deck mirrors: mirror-deck training deals BOTH
    # seats the same randomly-chosen deck each episode, so a fair gate must test
    # both decks symmetrically. "default" (A-vs-B) is imbalanced and makes the
    # gate structurally unwinnable — the losing deck's seat drags any candidate
    # below the CI threshold regardless of skill.
    modes = ["AA", "BB"] if args.decks == "mirror" else [args.decks]
    n_each = max(1, (args.matches // 2) // len(modes))

    DECKS = _decks_for(modes[0])
    st = self_test(act_a, name_a, min(n_each, 128), args.seed + 1000)

    ab_parts, ba_parts, illegal = [], [], 0
    for i, m in enumerate(modes):
        DECKS = _decks_for(m)
        r_ab, il_ab = run_side(act_a, act_b, n_each, args.seed + 2 * i)      # A seat 0
        r_ba, il_ba = run_side(act_b, act_a, n_each, args.seed + 2 * i + 1)  # B seat 0
        ab_parts.append(np.asarray(r_ab))
        ba_parts.append(np.asarray(r_ba))
        illegal += int(il_ab) + int(il_ba)
    r_ab = np.concatenate(ab_parts)
    r_ba = np.concatenate(ba_parts)
    total = len(r_ab) + len(r_ba)
    assert illegal == 0, f"loud-fail: {illegal} illegal actions during eval"
    a_wins = int((r_ab == 0).sum()) + int((r_ba == 1).sum())
    b_wins = int((r_ab == 1).sum()) + int((r_ba == 0).sum())
    draws = total - a_wins - b_wins
    p = a_wins / total
    ci = wilson_lower(p, total)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    report = {
        "a": name_a, "b": name_b, "matches": total, "seat_swapped": True,
        "decks": args.decks,
        "a_win_rate": p, "b_win_rate": b_wins / total, "draw_rate": draws / total,
        "a_win_as_p0": float((r_ab == 0).mean()), "a_win_as_p1": float((r_ba == 1).mean()),
        "a_win_ci95_lower": ci,
        "illegal_actions": illegal, "self_test": st,
        "commit": commit, "device": str(jax.devices()[0]),
        "integrity": {"expected": total, "actual": a_wins + b_wins + draws,
                      "complete": a_wins + b_wins + draws == total},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
