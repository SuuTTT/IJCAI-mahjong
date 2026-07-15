"""Gymnasium and PettingZoo adapters over the JAX core.

These are convenience wrappers for interactive/eval use; training at scale should
use boom.vec directly (vmapped, jit, no per-step host round-trips).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import engine, vec
from .engine import H, OBS_C, OBS_VEC, RESULT_DRAW, RESULT_ONGOING, TICKS_MAX, W

WIN_R, LOSE_R = 1.0, -1.0


def _np_obs(obs: engine.Obs) -> dict:
    return {"spatial": np.asarray(obs.spatial), "vector": np.asarray(obs.vector)}


class _Core:
    """Shared jitted callables (compiled once per process)."""

    step = staticmethod(jax.jit(engine.step))
    reset = staticmethod(jax.jit(engine.reset))
    observe = staticmethod(jax.jit(engine.observe, static_argnums=1))
    result = staticmethod(jax.jit(engine.result))
    flat_legal = staticmethod(jax.jit(vec.flat_legal, static_argnums=1))
    random_action = staticmethod(jax.jit(
        lambda k, s, p: vec.flat_to_triple(vec.random_legal_action(k, s, p)),
        static_argnums=2))


def _spaces():
    import gymnasium as gym

    obs_space = gym.spaces.Dict({
        "spatial": gym.spaces.Box(-np.inf, np.inf, (H, W, OBS_C), np.float32),
        "vector": gym.spaces.Box(-np.inf, np.inf, (OBS_VEC,), np.float32),
    })
    act_space = gym.spaces.Discrete(vec.N_ACTIONS)
    return obs_space, act_space


class BoomGymEnv:
    """Single-agent gymnasium.Env: you are player 0; the opponent is a policy
    `opponent(key, state) -> (2,3)-compatible player-1 action triple` (default:
    uniform random-legal). Reward: +1 win / -1 loss at terminal, else 0."""

    metadata = {"name": "boom/v1"}

    def __init__(self, opponent=None, decks=None, seed: int = 0):
        import gymnasium as gym  # noqa: F401  (import check)

        self.observation_space, self.action_space = _spaces()
        self._opponent = opponent
        self._decks = None if decks is None else jnp.asarray(decks)
        self._key = jax.random.PRNGKey(seed)
        self._state = None

    def _split(self):
        self._key, k = jax.random.split(self._key)
        return k

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._state = _Core.reset(self._split(), self._decks)
        obs = _np_obs(_Core.observe(self._state, 0))
        return obs, {"legal": np.asarray(_Core.flat_legal(self._state, 0))}

    def step(self, action: int):
        a0 = vec.flat_to_triple(jnp.int32(action))
        if self._opponent is None:
            a1 = _Core.random_action(self._split(), self._state, 1)
        else:
            a1 = jnp.asarray(self._opponent(self._split(), self._state), jnp.int32)
        self._state = _Core.step(self._state, jnp.stack([a0, a1]), None)
        r = int(_Core.result(self._state))
        terminated = r != RESULT_ONGOING
        reward = 0.0
        if terminated:
            reward = 0.0 if r == RESULT_DRAW else (WIN_R if r == 0 else LOSE_R)
        obs = _np_obs(_Core.observe(self._state, 0))
        info = {"legal": np.asarray(_Core.flat_legal(self._state, 0)),
                "illegal_count": np.asarray(self._state.illegal), "result": r}
        return obs, reward, terminated, False, info


class BoomParallelEnv:
    """PettingZoo ParallelEnv-style two-player adapter (players '0' and '1').
    Both agents receive player-centric obs/actions."""

    metadata = {"name": "boom/v1"}
    agents = possible_agents = ["0", "1"]

    def __init__(self, decks=None, seed: int = 0):
        self._decks = None if decks is None else jnp.asarray(decks)
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        obs_space, act_space = _spaces()
        self.observation_spaces = {a: obs_space for a in self.agents}
        self.action_spaces = {a: act_space for a in self.agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, k = jax.random.split(self._key)
        self._state = _Core.reset(k, self._decks)
        obs = {a: _np_obs(_Core.observe(self._state, int(a))) for a in self.agents}
        infos = {a: {"legal": np.asarray(_Core.flat_legal(self._state, int(a)))}
                 for a in self.agents}
        return obs, infos

    def step(self, actions: dict):
        triples = jnp.stack([vec.flat_to_triple(jnp.int32(actions["0"])),
                             vec.flat_to_triple(jnp.int32(actions["1"]))])
        self._state = _Core.step(self._state, triples, None)
        r = int(_Core.result(self._state))
        done = r != RESULT_ONGOING
        rew = {"0": 0.0, "1": 0.0}
        if done and r != RESULT_DRAW:
            rew["0"], rew["1"] = (WIN_R, LOSE_R) if r == 0 else (LOSE_R, WIN_R)
        obs = {a: _np_obs(_Core.observe(self._state, int(a))) for a in self.agents}
        term = {a: done for a in self.agents}
        trunc = {a: False for a in self.agents}
        infos = {a: {"legal": np.asarray(_Core.flat_legal(self._state, int(a))),
                     "result": r} for a in self.agents}
        return obs, rew, term, trunc, infos
