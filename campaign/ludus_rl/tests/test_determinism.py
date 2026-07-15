"""Determinism suite (AGENTS.md §1) — these are release blockers, never 'flaky'."""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boom import engine, vec

# full-length on the GPU runner; short on CPU where the scanned rollout's
# XLA compile alone runs tens of minutes — the determinism property is
# tick-count-independent
_CPU = jax.default_backend() == "cpu"
TICKS = 40 if _CPU else 300
BATCH = 2 if _CPU else 4


def state_hash(state: engine.State) -> str:
    h = hashlib.sha256()
    for field in state:
        h.update(np.ascontiguousarray(np.asarray(field)).tobytes())
    return h.hexdigest()


def _recorded_match(seed: int = 0):
    """One random-legal match: returns (key, action_log (T,2,3))."""
    key = jax.random.PRNGKey(seed)
    _, log = vec.rollout_random_jit(key, BATCH, TICKS, None, True)
    match_key = jax.random.split(key, BATCH)[0]
    return match_key, jnp.asarray(log)[:, 0]


def _trajectory_hashes(key, action_log, step_fn):
    state = engine.reset(key, None)
    hashes = []
    for t in range(action_log.shape[0]):
        state = step_fn(state, action_log[t], None)
        hashes.append(state_hash(jax.block_until_ready(state)))
    return hashes


@pytest.mark.skipif(jax.default_backend() == "cpu",
                    reason="eager full-match on CPU takes hours by nature; "
                           "jit≡eager is pinned on the GPU runner")
def test_jit_equals_nojit():
    # eager dispatch is ~1000x slower per tick on any backend; bit-identity is
    # per-op, so a short prefix proves the property without the hour-long run
    key, log = _recorded_match(0)
    log = log[:25]
    h_jit = _trajectory_hashes(key, log, jax.jit(engine.step))
    with jax.disable_jit():
        h_eager = _trajectory_hashes(key, log, engine.step)
    assert h_jit == h_eager, "jit vs no-jit state trajectories diverge"


@pytest.mark.skipif(jax.default_backend() == "cpu",
                    reason="replay loop hits per-call pjit cache-misses on CPU "
                           "(~14s/step; see docs/jax_cpu_regression_report.md) — "
                           "bit-identity is pinned on the GPU runner")
def test_replay_bit_identical():
    key, log = _recorded_match(1)
    h1 = state_hash(jax.block_until_ready(vec.replay_jit(key, log, None)))
    h2 = state_hash(jax.block_until_ready(vec.replay_jit(key, log, None)))
    h3 = state_hash(jax.block_until_ready(vec.replay(key, log, None)))
    assert h1 == h2 == h3, "replay from (seed, action_log) is not reproducible"


def test_reset_deterministic():
    k = jax.random.PRNGKey(7)
    assert state_hash(engine.reset(k, None)) == state_hash(engine.reset(k, None))


@pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()),
                    reason="needs a GPU to compare against CPU")
def test_cpu_equals_gpu():
    key, log = _recorded_match(2)
    cpu = jax.devices("cpu")[0]
    gpu = [d for d in jax.devices() if d.platform == "gpu"][0]
    hashes = {}
    for dev in (cpu, gpu):
        with jax.default_device(dev):
            k = jax.device_put(key, dev)
            lg = jax.device_put(log, dev)
            hashes[dev.platform] = _trajectory_hashes(k, lg, jax.jit(engine.step))
    assert hashes["cpu"] == hashes["gpu"], "CPU vs GPU state trajectories diverge"


@pytest.mark.skipif(jax.default_backend() == "cpu",
                    reason="replay loop hits per-call pjit cache-misses on CPU "
                           "(~14s/step; see docs/jax_cpu_regression_report.md) — "
                           "bit-identity is pinned on the GPU runner")
def test_vmapped_matches_single():
    """Match 0 of a vmapped rollout equals the same match run unbatched."""
    key = jax.random.PRNGKey(3)
    states, log = vec.rollout_random_jit(key, BATCH, TICKS, None, True)
    single = vec.replay_jit(jax.random.split(key, BATCH)[0], jnp.asarray(log)[:, 0], None)
    batched0 = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], states)
    for a, b in zip(batched0, single):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
