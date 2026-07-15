"""Warpath RL harness: observation, macro action space, scripted opponent, reward.

Warpath's raw action is a spatial, multi-command RTS interface (build->place,
per-unit orders, tile placement) — far too wide to learn from scratch. This
module exposes a compact, learnable MACRO layer on top of the pure-functional
engine:

  * obs(state, p)          -> (OBS_DIM,) float32 per-player feature vector
  * macro_action(state,p,m)-> (3,) int32 engine action (with auto-placement)
  * scripted_macro(state,p)-> a stationary rule_v0 opponent (returns a macro id)
  * reward(prev, cur, p)   -> scalar shaped + terminal reward

Everything is jit/vmap-friendly; p is a Python int (0/1) so both seats can share
one policy by swapping the view.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import engine as E

# ---- macro action space -------------------------------------------------
#  0            no-op
#  1..6         train unit type 0..5   (harvester, rifle, rocket, tank, bulldog, arty)
#  7..14        build building type 1..8 (power, barracks, factory, radar, lab,
#                                         turret, tower, refinery)
#  15           attack-move whole army at the ENEMY conyard
#  16           attack-move whole army home (defend)
#  17           attack-move whole army to map centre
#  18           rally point at the enemy conyard
N_ACT = 19
OBS_DIM = 47

_B_HP0 = float(E.B_HP[0])                 # conyard max HP (concrete, for norms)
_CON_TILE_X = jnp.array([5, 34], jnp.int32)
_CON_TILE_Y = jnp.array([15, 15], jnp.int32)
# placement ring around the conyard, cycled by building count (dir flips by seat)
_OFF_DX = jnp.array([2, 2, 2, 3, 3, 4, 4, 4], jnp.int32)
_OFF_DY = jnp.array([0, -2, 2, -4, 4, 0, -3, 3], jnp.int32)


def _pack(xt, yt):
    return jnp.clip(xt, 0, E.W - 1) * 64 + jnp.clip(yt, 0, E.H - 1)


def _own_bld_counts(state, p):
    valid = (state.b_type >= 0) & (state.b_owner == p) & (state.b_hp > 0)
    bt = jnp.where(valid, state.b_type, -1)
    return jnp.array([jnp.sum(bt == t) for t in range(9)], jnp.float32)


def _own_unit_counts(state, p):
    valid = (state.u_type >= 0) & (state.u_owner == p) & (state.u_hp > 0)
    ut = jnp.where(valid, state.u_type, -1)
    return jnp.array([jnp.sum(ut == t) for t in range(6)], jnp.float32)


def _con_frac(state, p):
    hp = jnp.sum(jnp.where((state.b_type == 0) & (state.b_owner == p) & (state.b_hp > 0),
                           state.b_hp, 0))
    return hp.astype(jnp.float32) / _B_HP0


def _centroid(state, p):
    m = (state.u_type >= 0) & (state.u_owner == p) & (state.u_hp > 0)
    n = jnp.maximum(jnp.sum(m), 1)
    cx = jnp.sum(jnp.where(m, state.u_x, 0)) / n / E.W_FP
    cy = jnp.sum(jnp.where(m, state.u_y, 0)) / n / E.H_FP
    return jnp.array([cx, cy], jnp.float32)


def obs(state, p: int) -> jax.Array:
    """Per-player feature vector; symmetric so both seats share a policy."""
    e = 1 - p
    scal = jnp.array([
        state.credits[p] / 5000.0, state.credits[e] / 5000.0,
        jnp.clip(E.power_balance(state, p) / 200.0, -1.0, 1.0),
        jnp.clip(E.power_balance(state, e) / 200.0, -1.0, 1.0),
        _con_frac(state, p), _con_frac(state, e),
        jnp.sum(jnp.where((state.u_owner == p) & (state.u_hp > 0), state.u_hp, 0)) / 8000.0,
        jnp.sum(jnp.where((state.u_owner == e) & (state.u_hp > 0), state.u_hp, 0)) / 8000.0,
        state.tick.astype(jnp.float32) / E.TIME_LIMIT,
        (state.pend_type[p] >= 0).astype(jnp.float32),
    ], jnp.float32)
    prod_busy = (state.prod_type[p] >= 0).astype(jnp.float32)            # (3,)
    return jnp.concatenate([
        scal,
        _own_bld_counts(state, p) / 3.0, _own_bld_counts(state, e) / 3.0,
        _own_unit_counts(state, p) / 8.0, _own_unit_counts(state, e) / 8.0,
        _centroid(state, p), _centroid(state, e), prod_busy,
    ])


def _place_tile(state, p: int):
    nb = jnp.sum((state.b_type >= 0) & (state.b_owner == p) & (state.b_hp > 0))
    i = jnp.mod(nb, 8)
    d = jnp.where(p == 0, 1, -1)
    xt = _CON_TILE_X[p] + d * _OFF_DX[i]
    yt = _CON_TILE_Y[p] + _OFF_DY[i]
    return _pack(xt, yt)


def macro_action(state, p: int, macro) -> jax.Array:
    """Map a macro id to a (3,) engine action. A ready pending building is
    auto-placed first (the agent never has to learn tile placement)."""
    con_e = _pack(_CON_TILE_X[1 - p], _CON_TILE_Y[1 - p])
    con_s = _pack(_CON_TILE_X[p], _CON_TILE_Y[p])
    centre = _pack(jnp.int32(E.W // 2), jnp.int32(E.H // 2))
    train = jnp.stack([jnp.int32(2), jnp.clip(macro - 1, 0, 5), jnp.int32(0)])
    build = jnp.stack([jnp.int32(1), jnp.clip(macro - 6, 1, 8), jnp.int32(0)])
    amove_e = jnp.array([3, 0, 0], jnp.int32).at[2].set(con_e)
    amove_s = jnp.array([3, 0, 0], jnp.int32).at[2].set(con_s)
    amove_c = jnp.array([3, 0, 0], jnp.int32).at[2].set(centre)
    rally_e = jnp.array([4, 0, 0], jnp.int32).at[2].set(con_e)
    noop = jnp.array([0, 0, 0], jnp.int32)
    act = jnp.select(
        [macro == 0,
         (macro >= 1) & (macro <= 6),
         (macro >= 7) & (macro <= 14),
         macro == 15, macro == 16, macro == 17, macro == 18],
        [noop, train, build, amove_e, amove_s, amove_c, rally_e], noop)
    # auto-place a finished building this tick, overriding the chosen macro
    ready = (state.pend_type[p] >= 0) & (state.pend_t[p] == 0)
    place = jnp.stack([jnp.int32(6), jnp.int32(0), _place_tile(state, p)])
    return jnp.where(ready, place, act)


def scripted_macro(state, p: int):
    """Stationary rule_v0 commander: power -> refinery -> barracks -> rifle
    stream, harvesters for economy, army waves at the enemy base."""
    om = E.owned_mask(state, p)
    cr = state.credits[p]
    have = lambda t: ((om >> t) & 1) > 0
    nharv = jnp.sum((state.u_type == 0) & (state.u_owner == p) & (state.u_hp > 0))
    nfight = jnp.sum((state.u_type >= 1) & (state.u_owner == p) & (state.u_hp > 0))
    # jnp.select = FIRST true condition wins (priority order, highest first)
    return jnp.select([
        (nfight >= 8) & (jnp.mod(state.tick, 60) < 3),   # 15 army wave at enemy base
        ~have(1) & (cr >= 800),                          #  7 power plant  (economy 1st)
        have(1) & ~have(8) & (cr >= 1500),               # 14 refinery -> free harvester
        have(8) & (nharv < 3) & (cr >= 1400),            #  1 train harvester
        have(1) & ~have(2) & (cr >= 500),                #  8 barracks
        have(2) & ~have(5) & (cr >= 2500),               # 11 turret (only when rich)
        have(2) & (cr >= 200),                           #  2 pump riflemen
    ], jnp.array([15, 7, 14, 1, 8, 11, 2], jnp.int32), default=jnp.int32(0))


def reward(prev, cur, p: int) -> jax.Array:
    """Terminal win/loss dominates; shaping rewards damaging the enemy conyard
    and penalises losing your own."""
    res = E.result(cur)
    term = 6.0 * (res == p).astype(jnp.float32) - 6.0 * (res == (1 - p)).astype(jnp.float32)
    e = 1 - p
    d_enemy = _con_frac(prev, e) - _con_frac(cur, e)      # enemy base HP lost (good)
    d_self = _con_frac(prev, p) - _con_frac(cur, p)       # own base HP lost (bad)
    return term + 3.0 * d_enemy - 3.0 * d_self
