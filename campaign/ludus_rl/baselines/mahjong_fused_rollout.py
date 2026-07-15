"""Fused, on-GPU env+policy rollout for the JAX MCR env (the issue-#3 payoff:
env steps AND kdens3 neural inference co-resident on the GPU, no host round-trips).

A single jitted ``lax.scan`` over T steps of B ``jax.vmap``-ed games. Each step:
  1. build (B,4) per-seat obs (B,4,38,4,9) and legal masks (B,4,235) from state,
  2. run the flax kdens3 policy as ONE batched GPU call over the 4*B rows,
  3. masked-argmax -> primary 235-code per seat; heuristic follow-discard (the
     env fuses a granted Peng/Chi claim with its follow discard),
  4. vmap ``jax_env.step`` -> next state.
Everything stays on-device; the only host touch the env ever makes is the fan
``pure_callback`` at a Hu event, which the env's random-legal mask omits, so
greedy kdens3 self-play here terminates by HUANG (wall exhaustion) fully on-GPU.

Self-play: all 4 seats share the frozen kdens3 net (a learner-vs-frozen split is
a trivial per-seat net swap; self-play is the cleanest thing to benchmark).

NEW file — imports mahjong/jax_env.py + baselines/mahjong_jax_policy.py, edits
neither.
"""
import sys, time, argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

sys.path.insert(0, "/root/ludus")
sys.path.insert(0, "/root/ludus/baselines")

from mahjong import jax_env as J
from mahjong_jax_policy import load_params, forward, N_ACT

NPZ = "/root/mcr_champion/kdens_s0_fp16.npz"


def rand_walls(B, seed=0):
    rng = np.random.default_rng(seed)
    base = np.repeat(np.arange(34, dtype=np.int8), 4)
    W = np.stack([rng.permutation(base) for _ in range(B)])
    Q = rng.integers(0, 4, size=B).astype(np.int32)
    return jnp.asarray(W), jnp.asarray(Q)


def _seat_obs(state):
    # (4,38,4,9) obs for the 4 seats of ONE game
    return jax.vmap(lambda s: J.obs(state, s))(jnp.arange(4))


def _aux_follow(state):
    # (4,) legal follow-discard per seat: most-held tile != live (== jax_env
    # random_action_fn aux; guaranteed a held, non-live tile).
    live = state.last_discard_tile
    hh = jnp.where(jnp.arange(34)[None, :] == live, 0, state.hands)
    return jnp.argmax(hh, axis=1).astype(jnp.int32)


def make_step_fn(bundle, dtype):
    """One fused (policy-forward + env-step) over a batch of B games."""
    def act(states, extra):
        # obs (B,4,38,4,9) ; mask (B,4,235) ; aux (B,4)
        obs = jax.vmap(_seat_obs)(states)
        mask = jax.vmap(J.legal_mask)(states)
        aux = jax.vmap(_aux_follow)(states)
        B = mask.shape[0]
        flat_obs = obs.reshape(B * 4, 38, 4, 9)
        flat_mask = mask.reshape(B * 4, N_ACT)
        logits = forward(bundle, flat_obs, flat_mask, dtype=dtype)   # ONE GPU call
        primary = jnp.argmax(logits, axis=1).astype(jnp.int32).reshape(B, 4)
        actions = jnp.stack([primary, aux], axis=2)                  # (B,4,2)
        nstates = jax.vmap(J.step)(states, actions)
        # carry a legality flag: chosen primary must be legal under the mask
        chosen_legal = jnp.take_along_axis(
            mask, primary[:, :, None], axis=2)[:, :, 0]              # (B,4) bool
        return nstates, chosen_legal.all()
    return act


def make_rollout(bundle, T, dtype):
    step_fn = make_step_fn(bundle, dtype)

    @jax.jit
    def rollout(walls, quans):
        states = jax.vmap(J.reset)(walls, quans)
        states, legal_flags = lax.scan(step_fn, states, None, length=T)
        return states, legal_flags.all()
    return rollout


def benchmark(B, T, bundle, dtype):
    roll = make_rollout(bundle, T, dtype)
    W, Q = rand_walls(B)
    # compile + warm
    st, ok = roll(W, Q)
    st.done.block_until_ready()
    t0 = time.time()
    st, ok = roll(W, Q)
    st.done.block_until_ready()
    dt = time.time() - t0
    done = int(np.asarray(st.done).sum())
    steps = B * T
    return dict(B=B, T=T, dt=dt, steps=steps, sps=steps / dt,
                policy_fwd_per_s=steps * 4 / dt,
                done=done, all_legal=bool(ok))


def correctness(B, T, bundle, dtype):
    """Run to (near-)terminal, assert 0 illegal actions, games reach done,
    hands stay non-negative, meld_cnt<=4, and report the ending mix."""
    roll = make_rollout(bundle, T, dtype)
    W, Q = rand_walls(B, seed=123)
    st, ok = roll(W, Q)
    st.done.block_until_ready()
    done = np.asarray(st.done)
    hands = np.asarray(st.hands)
    mc = np.asarray(st.meld_cnt)
    ending = np.asarray(st.ending)
    n_illegal = 0 if bool(ok) else 1
    return dict(
        B=B, T=T, all_legal=bool(ok), illegal=n_illegal,
        frac_done=float(done.mean()), n_done=int(done.sum()),
        hands_min=int(hands.min()), mc_max=int(mc.max()),
        ending_hist={int(e): int((ending == e).sum()) for e in np.unique(ending)},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bench", choices=["bench", "correct"])
    ap.add_argument("--batches", default="512,1024,4096")
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dt = {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}[args.dtype]
    bundle = load_params(NPZ)
    print("device:", jax.devices(), "dtype:", args.dtype)

    if args.mode == "correct":
        r = correctness(1024, 400, bundle, dt)
        print("CORRECTNESS:", r)
    else:
        rows = []
        for B in [int(x) for x in args.batches.split(",")]:
            try:
                r = benchmark(B, args.T, bundle, dt)
                rows.append(r)
                print(f"B={r['B']:5d} T={r['T']} dt={r['dt']:.3f}s  "
                      f"FUSED game-steps/s = {r['sps']:,.0f}  "
                      f"(policy-fwd/s = {r['policy_fwd_per_s']:,.0f})  "
                      f"done={r['done']}/{B} legal={r['all_legal']}")
            except Exception as e:
                print(f"B={B}: FAILED {type(e).__name__}: {str(e)[:200]}")
        if rows:
            best = max(rows, key=lambda r: r["sps"])
            print(f"\nBEST: B={best['B']} -> {best['sps']:,.0f} fused game-steps/s")
