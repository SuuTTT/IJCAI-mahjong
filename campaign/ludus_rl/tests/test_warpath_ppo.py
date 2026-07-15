"""Pin the pure PPO helper in baselines/warpath_ppo.py.

GAE is the advantage estimator; a sign/masking bug here is silent (training just
gets worse) so it earns an explicit numeric-reference test.
"""
import jax.numpy as jnp
import numpy as np

from baselines import warpath_ppo as W


def _ref_gae(rew, val, done, lastv, gamma, lam):
    """Textbook GAE, reverse recursion, done cuts bootstrap AND propagation."""
    T = len(rew)
    adv = [0.0] * T
    gae, nextv = 0.0, lastv
    for t in reversed(range(T)):
        delta = rew[t] + gamma * nextv * (1 - done[t]) - val[t]
        gae = delta + gamma * lam * (1 - done[t]) * gae
        adv[t] = gae
        nextv = val[t]
    return np.array(adv)


def _run(rew, val, done, lastv, gamma, lam):
    traj = {"rew": jnp.array(rew)[:, None],
            "val": jnp.array(val)[:, None],
            "done": jnp.array(done)[:, None]}
    adv, ret = W.gae(traj, jnp.array([lastv], jnp.float32), gamma, lam)
    return np.asarray(adv)[:, 0], np.asarray(ret)[:, 0]


def test_gae_matches_reference_no_done():
    rew, val, done = [1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [0.0, 0.0, 0.0]
    lastv, gamma, lam = 2.0, 0.9, 0.8
    adv, ret = _run(rew, val, done, lastv, gamma, lam)
    ref = _ref_gae(rew, val, done, lastv, gamma, lam)
    assert np.allclose(adv, ref, atol=1e-5), (adv, ref)
    # returns = advantage + value baseline
    assert np.allclose(ret, ref + np.array(val), atol=1e-5)


def test_gae_masks_at_done():
    # a terminal in the middle must not let value/advantage leak across the boundary
    rew, val = [1.0, 5.0, 2.0, 3.0], [0.5, 0.5, 0.5, 0.5]
    done = [0.0, 1.0, 0.0, 0.0]
    lastv, gamma, lam = 4.0, 0.99, 0.95
    adv, _ = _run(rew, val, done, lastv, gamma, lam)
    ref = _ref_gae(rew, val, done, lastv, gamma, lam)
    assert np.allclose(adv, ref, atol=1e-5), (adv, ref)
    # at the done step advantage collapses to (reward - value): no bootstrap, no future leak
    assert abs(adv[1] - (5.0 - 0.5)) < 1e-5, f"done step bootstrapped: adv={adv[1]}"


def test_gae_zero_when_value_is_exact():
    # if the critic is perfect (val == discounted return), advantages are ~0
    gamma, lam = 0.9, 1.0
    rew = [1.0, 1.0, 1.0]
    lastv = 0.0
    # exact values: V_t = sum_{k>=t} gamma^{k-t} r_k  (lastv=0, no bootstrap tail)
    val = [1 + 0.9 * (1 + 0.9 * 1), 1 + 0.9 * 1, 1.0]
    done = [0.0, 0.0, 0.0]
    adv, _ = _run(rew, val, done, lastv, gamma, lam)
    assert np.allclose(adv, 0.0, atol=1e-5), f"exact critic should give ~0 advantage, got {adv}"
