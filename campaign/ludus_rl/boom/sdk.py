"""boom.sdk — the four-call surface for Agent Challenge entrants.

Everything a policy needs, nothing else. See docs/CHALLENGE.md for the spec.

    from boom import sdk
    env = sdk.Env(seed=0)
    obs, mask = env.observe()          # your seat is always 0 (bottom)
    ...
    obs, mask, result = env.step(action)          # action: flat int or (s,x,y)

    sdk.submit_check("my_bot/params_latest.msgpack")   # validate before upload
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boom import engine, vec

N_ACTIONS = vec.N_ACTIONS
NOOP = 0


class Env:
    """Single-match wrapper: you are seat 0; the opponent is a callable
    (defaults to random-legal acting once per second)."""

    def __init__(self, seed: int = 0, opponent=None, decks=None):
        self._step = jax.jit(engine.step)
        self._observe = jax.jit(lambda s: engine.observe(s, 0))
        self._legal = jax.jit(lambda s: vec.flat_legal(s, 0))
        self._opp_random = jax.jit(
            lambda k, s: vec.flat_to_triple(vec.random_legal_action(k, s, 1)))
        self.state = engine.reset(jax.random.PRNGKey(seed), decks)
        self._key = jax.random.PRNGKey(seed ^ 0xBADCAB)
        self._opponent = opponent

    def observe(self):
        o = self._observe(self.state)
        return o, np.asarray(self._legal(self.state))

    def step(self, action):
        """action: flat int in [0, 2305) or a (slot, x, y) triple."""
        if np.ndim(action) == 0:
            a0 = np.asarray(vec.flat_to_triple(jnp.int32(action)))
        else:
            a0 = np.asarray(action, np.int32)
        tick = int(self.state.tick)
        if self._opponent is not None:
            a1 = np.asarray(self._opponent(self.state, tick), np.int32)
        elif tick % 5 == 0:
            self._key, k = jax.random.split(self._key)
            a1 = np.asarray(self._opp_random(k, self.state))
        else:
            a1 = np.asarray([4, 0, 0], np.int32)
        self.state = self._step(self.state,
                                jnp.asarray([a0, a1], jnp.int32), None)
        o, mask = self.observe()
        return o, mask, int(engine.result(self.state))


def load_policy(path: str | Path):
    """Load a submission msgpack into a callable (obs, mask, key) -> flat action."""
    from flax.serialization import from_bytes

    from baselines.ppo_selfplay import ActorCritic
    from boom.engine import H, OBS_C, OBS_VEC, W
    net = ActorCritic()
    tmpl = net.init(jax.random.PRNGKey(0),
                    jnp.zeros((1, H, W, OBS_C), jnp.float32),
                    jnp.zeros((1, OBS_VEC), jnp.float32))
    params = from_bytes(tmpl, Path(path).read_bytes())

    @jax.jit
    def act(obs_s, obs_v, mask, key):
        logits, _ = net.apply(params, obs_s[None], obs_v[None])
        return jax.random.categorical(key, jnp.where(mask, logits[0], -1e9) / 0.6)
    return lambda obs, mask, key: int(act(obs.spatial, obs.vector,
                                          jnp.asarray(mask), key))


def submit_check(path: str | Path, ticks: int = 150, seed: int = 7) -> dict:
    """Validate a submission exactly like the platform will: load it against the
    published architecture, then play it for `ticks` against random-legal and
    confirm it acts legally. Returns a report dict; raises on hard failure."""
    policy = load_policy(path)                     # raises if arch mismatch
    env = Env(seed=seed)
    key = jax.random.PRNGKey(seed)
    obs, mask = env.observe()
    plays, illegal = 0, 0
    result = -1
    for _ in range(ticks):
        key, k = jax.random.split(key)
        a = policy(obs, mask, k)
        if a != NOOP:
            plays += 1
            if not mask[a]:
                illegal += 1
        obs, mask, result = env.step(a)
        if result != -1:
            break
    report = {"ok": illegal == 0, "ticks": ticks, "card_plays": plays,
              "illegal": illegal, "result_reached": result,
              "size_bytes": Path(path).stat().st_size}
    if illegal:
        raise ValueError(f"policy played {illegal} illegal actions: {report}")
    return report
