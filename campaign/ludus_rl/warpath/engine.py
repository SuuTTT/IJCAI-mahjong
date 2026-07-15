"""Warpath v1 — RA2-class micro-RTS slice, deterministic JAX core (docs/11).

One-screen 1v1: ore economy -> production -> combat -> destroy the enemy
Construction Yard. Same contracts as Boom: int32 state, fixed-point board
(FP=16), pure functions, replay = (env_version, seed, action_log).

Actions per player per tick: [cmd, a, b] int32
  0 noop
  1 build building `a` at tile (a>>0 unused; b packs x*64+y)  -- see _do_build
  2 train unit type `a` (0 harvester, 1 rifleman, 2 rocketeer, 3 tank)
  3 army attack-move to tile (b packs x*64+y)
  4 set rally to tile (b packs x*64+y)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

ENV_VERSION = "warpath/v2"   # refinery/miner economy, forced-target, miner-hold, veterancy

# ---------------------------------------------------------------- board
W, H = 40, 30
FP = 16
W_FP, H_FP = W * FP, H * FP
MAX_U = 48                       # unit pool (both players)
MAX_B = 12                       # building pool
N_ORE = 6
TICKS_PER_S = 5
TIME_LIMIT = 3000                # 10 minutes

# ---------------------------------------------------------------- unit table
# RA2-accurate (Yuri's-Revenge-scaled): harvester=War Miner, rifleman=GI,
# rocketeer≈anti-armor infantry, tank=Grizzly, bulldog=Apocalypse-class.
#                harvester rifleman rocketeer tank   bulldog artillery
U_HP =    jnp.array([1000,  125,    125,      300,   800,   180], jnp.int32)
U_DMG =   jnp.array([0,     15,     45,       40,    90,    75], jnp.int32)
U_PERIOD = jnp.array([0,    4,      11,       8,     14,    22], jnp.int32)
U_RANGE_FP = jnp.array([0, int(4.0*FP), int(5.0*FP), int(4.75*FP), int(5.5*FP), int(8.0*FP)], jnp.int32)
U_SIGHT_FP = jnp.array([0, int(6.5*FP), int(7.0*FP), int(7.0*FP), int(7.5*FP), int(9.0*FP)], jnp.int32)
U_SPEED = jnp.array([9,    7,       5,        9,     6,     4], jnp.int32)
U_COST = jnp.array([1400,  200,     500,      700,   1750,  900], jnp.int32)
U_BUILD_T = jnp.array([80,  20,     45,       55,    130,   80], jnp.int32)
U_RADIUS_FP = jnp.array([int(0.6*FP), int(0.45*FP), int(0.45*FP),
                         int(0.6*FP), int(0.8*FP), int(0.55*FP)], jnp.int32)
U_MASS = jnp.array([8, 1, 1, 6, 12, 5], jnp.int32)
VET1, VET2 = 400, 1200          # damage-dealt XP thresholds -> veteran / elite
# AoE splash radius per unit (0 = single-target). Vehicles blast an area — a
# tank shell near a harvester wrecks it (RA2 crush + splash feel).
U_SPLASH_FP = jnp.array([0, 0, 0, int(1.1*FP), int(1.4*FP), int(2.2*FP)], jnp.int32)
# air layer: the rocketeer FLIES — ground-only units can't hit it, only units
# with anti-air (rocketeer itself + defense towers/turrets). It also ignores
# crush and ground collision.
U_AIR = jnp.array([0, 0, 1, 0, 0, 0], jnp.int32)
U_ANTIAIR = jnp.array([0, 0, 1, 0, 0, 0], jnp.int32)   # only rocketeers hit air (units)
B_ANTIAIR = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 0], jnp.int32)  # +refinery(8)
U_PREREQ = jnp.array([1 << 8, 0, 1 << 4, 0, 1 << 5, 1 << 4], jnp.int32)  # harvester<-refinery, artillery<-radar
U_VS = jnp.array([[0, 0, 0],
                  [100, 55, 60],
                  [70, 140, 110],
                  [55, 100, 130],
                  [70, 120, 140],
                  [110, 90, 150]], jnp.int32)   # artillery: anti-infantry+siege AoE

# ---------------------------------------------------------------- buildings
#                 conyard power barracks factory radar lab   turret tower refinery
B_HP = jnp.array([1000,   750,  800,     1000,  1000, 800,  600,   700, 900], jnp.int32)
B_COST = jnp.array([0,    800,  500,     2000,  1000, 2000, 600,   1500, 1500], jnp.int32)
B_BUILD_T = jnp.array([0, 50,   40,      90,    70,   100,  40,    70, 60], jnp.int32)
# defense turrets auto-fire: dmg/range/period (0 = non-combat building)
B_DMG = jnp.array([0, 0, 0, 0, 0, 0, 35, 90, 0], jnp.int32)
B_RANGE_FP = jnp.array([0, 0, 0, 0, 0, 0, int(6.0*FP), int(7.0*FP), 0], jnp.int32)
B_PERIOD = jnp.array([0, 0, 0, 0, 0, 0, 6, 12, 0], jnp.int32)
# RA2-style prerequisite graph as bitmasks over OWNED building types:
# power<-conyard; barracks<-power; factory<-power+barracks; radar<-power;
# lab<-factory+radar. Dynamic by construction: checks run against buildings
# alive RIGHT NOW, so losing the lab re-locks lab tech automatically.
B_PREREQ = jnp.array([0, 1 << 0, 1 << 1, (1 << 1) | (1 << 2),
                      1 << 1, (1 << 3) | (1 << 4),
                      1 << 2, 1 << 5, 1 << 1], jnp.int32)  # turret<-barracks, tower<-lab, refinery<-power
B_POWER = jnp.array([0, 100, -10, -25, -50, -60, -20, -75, -30], jnp.int32)
BLD_NEAR_FP = 6 * FP            # new buildings must be near an own building

ORE_X = jnp.array([8, 8, 12, 28, 32, 32], jnp.int32) * FP + FP // 2
ORE_Y = jnp.array([7, 23, 15, 15, 7, 23], jnp.int32) * FP + FP // 2
ORE_START = 6000                # credits per patch
ORE_LOAD = 300                  # per trip
MINE_T = 20                     # ticks to fill
UNLOAD_T = 10

CON_X = jnp.array([5, 34], jnp.int32) * FP + FP // 2
CON_Y = jnp.array([15, 15], jnp.int32) * FP + FP // 2
START_CREDITS = 5000        # RA2 skirmish scale (affords the tree, not trivial)


class State(NamedTuple):
    tick: jax.Array
    credits: jax.Array           # (2,)
    # units
    u_type: jax.Array            # (MAX_U,) -1 empty
    u_owner: jax.Array
    u_hp: jax.Array
    u_x: jax.Array
    u_y: jax.Array
    u_cd: jax.Array
    u_vet: jax.Array             # accumulated combat XP; ranks up at thresholds
    u_cargo: jax.Array           # harvester: 0 empty .. ORE_LOAD full; timers piggyback
    u_mode: jax.Array            # harvester: 0 to-ore 1 mining 2 to-base 3 unloading
    u_ax: jax.Array              # attack-move goal (fp; -1 = none)
    u_ay: jax.Array
    u_tgt: jax.Array             # forced target: -1 none, 0..MAX_U-1 unit, +MAX_U building
    # buildings
    b_type: jax.Array            # (MAX_B,) -1 empty
    b_owner: jax.Array
    b_hp: jax.Array
    b_x: jax.Array
    b_y: jax.Array
    b_prog: jax.Array            # construction ticks remaining (0 = operational)
    b_cd: jax.Array              # defense-turret weapon cooldown
    # production: one queue per (player, source) 0=barracks 1=factory 2=conyard(harv)
    prod_type: jax.Array         # (2,3) -1 idle (current item)
    prod_t: jax.Array            # (2,3) remaining ticks
    prod_q: jax.Array            # (2,3,8) queued unit types, -1 empty (RA2 queue)
    pend_type: jax.Array         # (2,) building constructing in the sidebar, -1 none
    pend_t: jax.Array            # (2,) ticks until the pending building is READY
    rally_x: jax.Array           # (2,)
    rally_y: jax.Array
    ore: jax.Array               # (N_ORE,)
    illegal: jax.Array           # (2,)


def _isqrt(x):
    """Integer sqrt via float — inputs < 2**24 so float32 is exact enough."""
    return jnp.sqrt(x.astype(jnp.float32)).astype(jnp.int32)


def _d2(ax, ay, bx, by):
    dx = (ax - bx) // 4
    dy = (ay - by) // 4
    return dx * dx + dy * dy


def _r2(r_fp):
    r = r_fp // 4
    return r * r


def reset(key: jax.Array | None = None) -> State:
    del key
    zu = jnp.zeros(MAX_U, jnp.int32)
    zb = jnp.zeros(MAX_B, jnp.int32)
    u_type = jnp.full(MAX_U, -1, jnp.int32)
    u_owner, u_hp = zu, zu
    # one starting harvester each (slots 0, 1)
    # RA2: no starting harvester — the Refinery grants a free one when built
    u_x, u_y = zu, zu
    b_type = jnp.full(MAX_B, -1, jnp.int32)
    b_type = b_type.at[0].set(0).at[1].set(0)
    b_owner = zb.at[1].set(1)
    b_hp = zb.at[0].set(B_HP[0]).at[1].set(B_HP[0])
    b_x = zb.at[0].set(CON_X[0]).at[1].set(CON_X[1])
    b_y = zb.at[0].set(CON_Y[0]).at[1].set(CON_Y[1])
    return State(
        tick=jnp.int32(0), credits=jnp.full(2, START_CREDITS, jnp.int32),
        u_type=u_type, u_owner=u_owner, u_hp=u_hp, u_x=u_x, u_y=u_y,
        u_cd=zu, u_vet=zu, u_cargo=zu, u_mode=zu,
        u_ax=jnp.full(MAX_U, -1, jnp.int32), u_ay=jnp.full(MAX_U, -1, jnp.int32),
        u_tgt=jnp.full(MAX_U, -1, jnp.int32),
        b_type=b_type, b_owner=b_owner, b_hp=b_hp, b_x=b_x, b_y=b_y,
        b_prog=jnp.zeros(MAX_B, jnp.int32), b_cd=jnp.zeros(MAX_B, jnp.int32),
        prod_type=jnp.full((2, 3), -1, jnp.int32), prod_t=jnp.zeros((2, 3), jnp.int32),
        prod_q=jnp.full((2, 3, 8), -1, jnp.int32),
        pend_type=jnp.full(2, -1, jnp.int32), pend_t=jnp.zeros(2, jnp.int32),
        rally_x=jnp.array([CON_X[0] + 4 * FP, CON_X[1] - 4 * FP], jnp.int32),
        rally_y=jnp.array([CON_Y[0], CON_Y[1]], jnp.int32),
        ore=jnp.full(N_ORE, ORE_START, jnp.int32),
        illegal=jnp.zeros(2, jnp.int32))


# ---------------------------------------------------------------- actions
def _unpack_xy(b):
    x = jnp.clip(b // 64, 0, W - 1) * FP + FP // 2
    y = jnp.clip(b % 64, 0, H - 1) * FP + FP // 2
    return x, y


def _do_build(state: State, p, btype, bx, by):
    """cmd 1: START a sidebar construction (RA2). Deduct cost, run pend_t; the
    finished building is PLACED later via cmd 6. Only one pending at a time."""
    btype = jnp.clip(btype, 1, 8)                     # conyard not buildable
    afford = state.credits[p] >= B_COST[btype]
    only_one = ~jnp.any((state.b_type == btype) & (state.b_owner == p) & (state.b_hp > 0))
    om = owned_mask(state, p)
    prereq_ok = (om & B_PREREQ[btype]) == B_PREREQ[btype]
    idle = state.pend_type[p] < 0
    ok = afford & only_one & prereq_ok & idle
    return state._replace(
        credits=jnp.where(ok, state.credits.at[p].add(-B_COST[btype]), state.credits),
        pend_type=jnp.where(ok, state.pend_type.at[p].set(btype), state.pend_type),
        pend_t=jnp.where(ok, state.pend_t.at[p].set(B_BUILD_T[btype]), state.pend_t),
        illegal=jnp.where(ok, state.illegal, state.illegal.at[p].add(1))), ok


def _placeable(state, p, btype, bx, by):
    """Is (bx,by) a legal spot: near own building, not overlapping a building."""
    near_own = jnp.any((state.b_type >= 0) & (state.b_owner == p) & (state.b_hp > 0)
                       & (_d2(state.b_x, state.b_y, bx, by) <= _r2(jnp.int32(BLD_NEAR_FP))))
    clear = ~jnp.any((state.b_type >= 0) & (state.b_hp > 0)
                     & (_d2(state.b_x, state.b_y, bx, by) <= _r2(jnp.int32(2 * FP))))
    on_map = (bx >= FP) & (bx < W_FP - FP) & (by >= FP) & (by < H_FP - FP)
    return near_own & clear & on_map


def _do_place(state: State, p, bx, by):
    """cmd 6: place the READY pending building at (bx,by), operational at once."""
    btype = state.pend_type[p]
    free = jnp.argmax(state.b_type == -1)
    has_free = state.b_type[free] == -1
    ready = (btype >= 0) & (state.pend_t[p] == 0)
    ok = ready & has_free & _placeable(state, p, btype, bx, by)
    bt = jnp.maximum(btype, 0)
    # RA2: a Refinery (type 8) comes with a free harvester
    gives_harv = ok & (bt == 8)
    ufree = jnp.argmax(state.u_hp <= 0)
    has_ufree = state.u_hp[ufree] <= 0
    spawn = gives_harv & has_ufree
    state = state._replace(
        u_type=jnp.where(spawn, state.u_type.at[ufree].set(0), state.u_type),
        u_owner=jnp.where(spawn, state.u_owner.at[ufree].set(p), state.u_owner),
        u_hp=jnp.where(spawn, state.u_hp.at[ufree].set(U_HP[0]), state.u_hp),
        u_x=jnp.where(spawn, state.u_x.at[ufree].set(bx), state.u_x),
        u_y=jnp.where(spawn, state.u_y.at[ufree].set(by + 2 * FP), state.u_y),
        u_mode=jnp.where(spawn, state.u_mode.at[ufree].set(0), state.u_mode),
        u_cargo=jnp.where(spawn, state.u_cargo.at[ufree].set(0), state.u_cargo),
        u_ax=jnp.where(spawn, state.u_ax.at[ufree].set(-1), state.u_ax),
        u_ay=jnp.where(spawn, state.u_ay.at[ufree].set(-1), state.u_ay),
        u_tgt=jnp.where(spawn, state.u_tgt.at[ufree].set(-1), state.u_tgt))
    return state._replace(
        b_type=jnp.where(ok, state.b_type.at[free].set(bt), state.b_type),
        b_owner=jnp.where(ok, state.b_owner.at[free].set(p), state.b_owner),
        b_hp=jnp.where(ok, state.b_hp.at[free].set(B_HP[bt]), state.b_hp),
        b_prog=jnp.where(ok, state.b_prog.at[free].set(0), state.b_prog),
        b_x=jnp.where(ok, state.b_x.at[free].set(bx), state.b_x),
        b_y=jnp.where(ok, state.b_y.at[free].set(by), state.b_y),
        pend_type=jnp.where(ok, state.pend_type.at[p].set(-1), state.pend_type),
        illegal=jnp.where(ok, state.illegal, state.illegal.at[p].add(1)))


def _has_bldg(state, p, btype):
    return jnp.any((state.b_type == btype) & (state.b_owner == p)
                   & (state.b_hp > 0) & (state.b_prog == 0))


def owned_mask(state, p):
    """Bitmask of building types the player owns OPERATIONAL (prereq input)."""
    own = ((state.b_type >= 0) & (state.b_owner == p) & (state.b_hp > 0)
           & (state.b_prog == 0))
    types = jnp.arange(9)
    present = jnp.any(own[None, :] & (state.b_type[None, :] == types[:, None]), axis=1)
    return jnp.sum(jnp.where(present, 1 << types, 0)).astype(jnp.int32)


def power_balance(state, p):
    own = ((state.b_type >= 0) & (state.b_owner == p) & (state.b_hp > 0)
           & (state.b_prog == 0))
    return jnp.sum(jnp.where(own, B_POWER[jnp.maximum(state.b_type, 0)], 0))


def _do_train(state: State, p, utype):
    utype = jnp.clip(utype, 0, 4)
    src = jnp.where(utype == 0, 2,
                    jnp.where((utype == 1) | (utype == 2), 0, 1))
    src_bldg = jnp.where(src == 2, 0, jnp.where(src == 0, 2, 3))
    has_src = _has_bldg(state, p, src_bldg)
    idle = state.prod_type[p, src] == -1
    afford = state.credits[p] >= U_COST[utype]
    om = owned_mask(state, p)
    prereq_ok = (om & U_PREREQ[utype]) == U_PREREQ[utype]
    # LOW POWER (consumption exceeds production) slows production 2x
    powered = power_balance(state, p) >= 0
    t = jnp.where(powered, U_BUILD_T[utype], U_BUILD_T[utype] * 2)
    q = state.prod_q[p, src]
    slot = jnp.argmax(q == -1)                    # first free queue slot
    has_slot = q[slot] == -1
    # start_now charges (build begins); enqueue is FREE until it starts (RA2:
    # money isn't spent until a queued item actually begins building)
    start_now = has_src & idle & afford & prereq_ok
    enqueue = has_src & ~idle & prereq_ok & has_slot
    ok = start_now | enqueue
    return state._replace(
        credits=jnp.where(start_now, state.credits.at[p].add(-U_COST[utype]),
                          state.credits),
        prod_type=jnp.where(start_now, state.prod_type.at[p, src].set(utype),
                            state.prod_type),
        prod_t=jnp.where(start_now, state.prod_t.at[p, src].set(t), state.prod_t),
        prod_q=jnp.where(enqueue, state.prod_q.at[p, src, slot].set(utype),
                         state.prod_q),
        illegal=jnp.where(ok, state.illegal, state.illegal.at[p].add(1)))


def _process_action(state: State, p, action):
    cmd, a, b = action[0], action[1], action[2]
    x, y = _unpack_xy(b)
    st_b, _ = _do_build(state, p, a, x, y)
    st_t = _do_train(state, p, a)
    combat = (state.u_type >= 1) & (state.u_owner == p) & (state.u_hp > 0)
    # attack-move to a position clears any forced target (a ground order overrides)
    st_am = state._replace(u_ax=jnp.where(combat, x, state.u_ax),
                           u_ay=jnp.where(combat, y, state.u_ay),
                           u_tgt=jnp.where(combat, -1, state.u_tgt))
    st_r = state._replace(rally_x=state.rally_x.at[p].set(x),
                          rally_y=state.rally_y.at[p].set(y))
    # cmd 5: single-unit attack-move order (unit id in `a`) — selection control
    uid = jnp.clip(a, 0, MAX_U - 1)
    mine = (state.u_owner[uid] == p) & (state.u_hp[uid] > 0) & (state.u_type[uid] >= 0)
    st_u = state._replace(u_ax=jnp.where(mine, state.u_ax.at[uid].set(x), state.u_ax),
                          u_ay=jnp.where(mine, state.u_ay.at[uid].set(y), state.u_ay),
                          u_tgt=jnp.where(mine, state.u_tgt.at[uid].set(-1), state.u_tgt))
    # cmd 8: force-attack — lock unit `a` onto a specific entity `b`
    #   (0..MAX_U-1 = enemy unit, +MAX_U = enemy building), overriding auto-acquire
    ftgt = jnp.clip(b, 0, MAX_U + MAX_B - 1)
    st_ft = state._replace(
        u_tgt=jnp.where(mine, state.u_tgt.at[uid].set(ftgt), state.u_tgt),
        u_ax=jnp.where(mine, state.u_ax.at[uid].set(-1), state.u_ax),
        u_ay=jnp.where(mine, state.u_ay.at[uid].set(-1), state.u_ay))
    st_p = _do_place(state, p, x, y)                  # cmd 6: place ready building
    # cmd 7: cancel the pending sidebar construction, refund its cost
    has_pend = state.pend_type[p] >= 0
    refund = jnp.where(has_pend, B_COST[jnp.maximum(state.pend_type[p], 0)], 0)
    st_c = state._replace(
        credits=state.credits.at[p].add(refund),
        pend_type=state.pend_type.at[p].set(-1),
        pend_t=state.pend_t.at[p].set(0))
    return jax.tree.map(
        lambda n, bb, tt, am, rr, uu, pp, cc, ff: jnp.select(
            [cmd == 1, cmd == 2, cmd == 3, cmd == 4, cmd == 5, cmd == 6, cmd == 7,
             cmd == 8],
            [bb, tt, am, rr, uu, pp, cc, ff], n),
        state, st_b, st_t, st_am, st_r, st_u, st_p, st_c, st_ft)


# ---------------------------------------------------------------- production
def _tick_production(state: State) -> State:
    # cancel any active/queued item whose prerequisite building was lost
    # (RA2: destroy the radar and its queued rocketeers stop). No refund.
    om = jnp.stack([owned_mask(state, 0), owned_mask(state, 1)])   # (2,)
    def prereq_ok(ut):
        pre = U_PREREQ[jnp.maximum(ut, 0)]
        return (om[:, None] & pre) == pre                         # (2,3) broadcast
    cur_ok = (state.prod_type < 0) | prereq_ok(state.prod_type)
    state = state._replace(
        prod_type=jnp.where(cur_ok, state.prod_type, -1),
        prod_t=jnp.where(cur_ok, state.prod_t, 0))
    # prune queued entries that lost their prereq (shift survivors forward)
    q = state.prod_q                                              # (2,3,8)
    q_ok = (q < 0) | ((om[:, None, None] & U_PREREQ[jnp.maximum(q, 0)])
                      == U_PREREQ[jnp.maximum(q, 0)])
    state = state._replace(prod_q=jnp.where(q_ok, q, -1))
    running = state.prod_type >= 0
    t = jnp.where(running, jnp.maximum(state.prod_t - 1, 0), state.prod_t)
    done = running & (t == 0)

    def spawn_one(state, p, src):
        ut = state.prod_type[p, src]
        fire = done[p, src]
        free = jnp.argmax(state.u_type == -1)
        has_free = state.u_type[free] == -1
        ok = fire & has_free
        sx = CON_X[p] + jnp.where(p == 0, 3 * FP, -3 * FP)
        sy = CON_Y[p] + (src - 1) * 2 * FP
        harv = ut == 0
        return state._replace(
            u_type=jnp.where(ok, state.u_type.at[free].set(ut), state.u_type),
            u_owner=jnp.where(ok, state.u_owner.at[free].set(p), state.u_owner),
            u_hp=jnp.where(ok, state.u_hp.at[free].set(U_HP[jnp.maximum(ut, 0)]), state.u_hp),
            u_x=jnp.where(ok, state.u_x.at[free].set(sx), state.u_x),
            u_y=jnp.where(ok, state.u_y.at[free].set(sy), state.u_y),
            u_cargo=jnp.where(ok, state.u_cargo.at[free].set(0), state.u_cargo),
            u_mode=jnp.where(ok, state.u_mode.at[free].set(0), state.u_mode),
            u_ax=jnp.where(ok & ~harv, state.u_ax.at[free].set(state.rally_x[p]),
                           jnp.where(ok, state.u_ax.at[free].set(-1), state.u_ax)),
            u_ay=jnp.where(ok & ~harv, state.u_ay.at[free].set(state.rally_y[p]),
                           jnp.where(ok, state.u_ay.at[free].set(-1), state.u_ay)),
            u_tgt=jnp.where(ok, state.u_tgt.at[free].set(-1), state.u_tgt),
            u_cd=jnp.where(ok, state.u_cd.at[free].set(0), state.u_cd))

    for p in range(2):
        for src in range(3):
            state = spawn_one(state, p, src)
    # RA2 queue: a finished item pulls the next queued type — and charges its
    # cost NOW (money is spent when the item begins, not when queued). If the
    # player can't afford it yet, it waits in the queue.
    nxt = state.prod_q[:, :, 0]
    cost_next = U_COST[jnp.maximum(nxt, 0)]          # (2,3)
    afford_next = state.credits[:, None] >= cost_next
    has_next = done & (nxt >= 0) & afford_next
    powered = jnp.stack([power_balance(state, 0) >= 0,
                         power_balance(state, 1) >= 0])[:, None]
    nxt_t = jnp.where(powered, U_BUILD_T[jnp.maximum(nxt, 0)],
                      U_BUILD_T[jnp.maximum(nxt, 0)] * 2)
    new_type = jnp.where(has_next, nxt, jnp.where(done, -1, state.prod_type))
    new_t = jnp.where(has_next, nxt_t, t)
    shifted = jnp.concatenate([state.prod_q[:, :, 1:],
                               jnp.full((2, 3, 1), -1, jnp.int32)], axis=2)
    new_q = jnp.where(has_next[:, :, None], shifted, state.prod_q)
    # deduct the started items' cost (sum per player across sources)
    spend = jnp.sum(jnp.where(has_next, cost_next, 0), axis=1)
    new_credits = state.credits - spend
    return state._replace(prod_t=new_t, prod_type=new_type, prod_q=new_q,
                          credits=new_credits)


# ---------------------------------------------------------------- harvesters
def _step_toward(x, y, tx, ty, speed):
    dx, dy = tx - x, ty - y
    dist = jnp.maximum(_isqrt(_d2(x, y, tx, ty)) * 4, 1)
    step = jnp.minimum(speed, dist)
    return (x + dx * step // dist, y + dy * step // dist)


def _tick_harvesters(state: State) -> State:
    is_h = (state.u_type == 0) & (state.u_hp > 0)
    # nearest live ore patch per harvester
    d2o = _d2(state.u_x[:, None], state.u_y[:, None], ORE_X[None, :], ORE_Y[None, :])
    d2o = jnp.where((state.ore > 0)[None, :], d2o, jnp.int32(2**30))
    near_ore = jnp.argmin(d2o, axis=1)
    ore_x, ore_y = ORE_X[near_ore], ORE_Y[near_ore]
    at_ore = _d2(state.u_x, state.u_y, ore_x, ore_y) <= _r2(jnp.int32(FP))
    base_x, base_y = CON_X[state.u_owner], CON_Y[state.u_owner]
    at_base = _d2(state.u_x, state.u_y, base_x, base_y) <= _r2(jnp.int32(2 * FP))

    mode = state.u_mode
    cargo = state.u_cargo
    # transitions
    mode = jnp.where(is_h & (mode == 0) & at_ore, 1, mode)
    mining_done = is_h & (mode == 1) & (cargo >= MINE_T)
    mode = jnp.where(mining_done, 2, mode)
    mode = jnp.where(is_h & (mode == 2) & at_base, 3, mode)
    unload_done = is_h & (mode == 3) & (cargo <= 0)
    mode = jnp.where(unload_done, 0, mode)
    # timers: cargo counts up while mining, down while unloading
    mine_gain = is_h & (mode == 1)
    ore_left = state.ore[near_ore] > 0
    cargo = jnp.where(mine_gain & ore_left, cargo + 1, cargo)
    ore = state.ore.at[near_ore].add(
        jnp.where(mine_gain & ore_left, -(ORE_LOAD // MINE_T), 0))
    unloading = is_h & (mode == 3)
    cargo = jnp.where(unloading, jnp.maximum(cargo - 2, 0), cargo)
    pay = jnp.where(unloading, jnp.minimum(cargo, 2) * (ORE_LOAD // MINE_T) // 2, 0)
    credits = state.credits.at[0].add(jnp.sum(jnp.where(state.u_owner == 0, pay, 0)))
    credits = credits.at[1].add(jnp.sum(jnp.where(state.u_owner == 1, pay, 0)))
    # forced target (cmd 8): the War-Miner chases an enemy UNIT to ram/crush it,
    # then HOLDS in place (mode 4) waiting for the next order — it does NOT drift
    # back to mining on its own. Building targets are ignored for harvesters.
    ft = state.u_tgt
    ft_u = jnp.clip(ft, 0, MAX_U - 1)
    ft_alive = ((ft >= 0) & (ft < MAX_U) & (state.u_hp[ft_u] > 0)
                & (state.u_type[ft_u] >= 0) & (state.u_owner[ft_u] != state.u_owner))
    chasing = is_h & ft_alive
    target_died = is_h & (ft >= 0) & ~ft_alive        # ordered target gone -> hold
    # movement priority: chase > manual move > auto harvest cycle
    manual = is_h & (state.u_ax >= 0) & ~chasing
    arrived = manual & (_d2(state.u_x, state.u_y, state.u_ax, state.u_ay)
                        <= _r2(jnp.int32(FP)))
    on_ore = _d2(state.u_x, state.u_y, ore_x, ore_y) <= _r2(jnp.int32(2 * FP))
    ftx, fty = state.u_x[ft_u], state.u_y[ft_u]
    tx = jnp.where(chasing, ftx,
                   jnp.where(manual, state.u_ax, jnp.where(mode == 2, base_x, ore_x)))
    ty = jnp.where(chasing, fty,
                   jnp.where(manual, state.u_ay, jnp.where(mode == 2, base_y, ore_y)))
    # mode 4 = HOLD (idle after a manual order or a finished crush; no auto-mining)
    moving = is_h & (chasing | manual | (mode == 0) | (mode == 2))
    nx, ny = _step_toward(state.u_x, state.u_y, tx, ty, U_SPEED[0])
    new_ax = jnp.where(arrived, -1, state.u_ax)
    new_ay = jnp.where(arrived, -1, state.u_ay)
    # while manually moving, suppress mining; on arrival: landed on ore -> rejoin
    # the harvest cycle, landed elsewhere -> HOLD. Chasing / finished-crush -> HOLD.
    mode = jnp.where(manual & ~arrived, 0, mode)
    mode = jnp.where(arrived & on_ore, 0, mode)
    mode = jnp.where(arrived & ~on_ore, 4, mode)
    mode = jnp.where(chasing | target_died, 4, mode)
    new_tgt = jnp.where(is_h, jnp.where(ft_alive, ft, -1), state.u_tgt)
    return state._replace(
        u_mode=mode, u_cargo=cargo, ore=jnp.maximum(ore, 0), credits=credits,
        u_tgt=new_tgt,
        u_ax=jnp.where(is_h, new_ax, state.u_ax),
        u_ay=jnp.where(is_h, new_ay, state.u_ay),
        u_x=jnp.where(moving, nx, state.u_x), u_y=jnp.where(moving, ny, state.u_y))


# ---------------------------------------------------------------- combat
def _tick_combat(state: State) -> State:
    utype = jnp.maximum(state.u_type, 0)
    fighter = (state.u_type >= 1) & (state.u_hp > 0)
    # nearest enemy unit / building within sight
    d2u = _d2(state.u_x[:, None], state.u_y[:, None],
              state.u_x[None, :], state.u_y[None, :])
    vic_air = U_AIR[jnp.maximum(state.u_type, 0)]
    can_aa = U_ANTIAIR[utype]                       # firer can hit air?
    air_ok = (vic_air[None, :] == 0) | (can_aa[:, None] == 1)
    elig_u = (fighter[:, None] & (state.u_hp > 0)[None, :]
              & (state.u_owner[:, None] != state.u_owner[None, :])
              & (state.u_type[None, :] >= 0) & air_ok)
    d2b = _d2(state.u_x[:, None], state.u_y[:, None],
              state.b_x[None, :], state.b_y[None, :])
    elig_b = (fighter[:, None] & (state.b_hp > 0)[None, :]
              & (state.b_type >= 0)[None, :]
              & (state.u_owner[:, None] != state.b_owner[None, :]))
    sight2 = _r2(U_SIGHT_FP[utype])[:, None]
    BIG = jnp.int32(2**30)
    cand = jnp.concatenate([jnp.where(elig_u & (d2u <= sight2), d2u, BIG),
                            jnp.where(elig_b & (d2b <= sight2), d2b, BIG)], axis=1)
    best = jnp.argmin(cand, axis=1)
    has_tgt = cand[jnp.arange(MAX_U), best] < BIG
    # forced target (cmd 8) overrides auto-acquire and ignores sight range: the
    # unit locks onto the ordered entity until it dies, then reverts to auto. This
    # is what lets you attack a building behind a screen of enemy units.
    ft = state.u_tgt
    ft_is_b = ft >= MAX_U
    ft_u = jnp.clip(ft, 0, MAX_U - 1)
    ft_b = jnp.clip(ft - MAX_U, 0, MAX_B - 1)
    ft_u_ok = ((~ft_is_b) & (ft >= 0) & (state.u_hp[ft_u] > 0)
               & (state.u_type[ft_u] >= 0)
               & (state.u_owner[ft_u] != state.u_owner)
               & ((U_AIR[jnp.maximum(state.u_type[ft_u], 0)] == 0) | (can_aa == 1)))
    ft_b_ok = (ft_is_b & (state.b_hp[ft_b] > 0) & (state.b_type[ft_b] >= 0)
               & (state.b_owner[ft_b] != state.u_owner))
    ft_valid = fighter & (ft >= 0) & (ft_u_ok | ft_b_ok)
    best = jnp.where(ft_valid, ft, best)
    has_tgt = has_tgt | ft_valid
    tgt_is_b = best >= MAX_U
    bi = jnp.clip(best - MAX_U, 0, MAX_B - 1)
    ui = jnp.clip(best, 0, MAX_U - 1)
    tx = jnp.where(tgt_is_b, state.b_x[bi], state.u_x[ui])
    ty = jnp.where(tgt_is_b, state.b_y[bi], state.u_y[ui])
    d2t = _d2(state.u_x, state.u_y, tx, ty)
    in_range = d2t <= _r2(U_RANGE_FP[utype])

    # damage with class multipliers
    vs_class = jnp.where(tgt_is_b, 2, jnp.where(state.u_type[ui] == 0, 1,
                         jnp.where(state.u_type[ui] >= 3, 1, 0)))
    mult = U_VS[utype, vs_class]
    can_fire = fighter & has_tgt & in_range & (state.u_cd == 0)
    # veterancy: 0=rookie 1=veteran(+25% dmg) 2=elite(+50% dmg). XP accrues from
    # damage dealt; ranks up at VET1/VET2 thresholds.
    rank = (state.u_vet >= VET1).astype(jnp.int32) + (state.u_vet >= VET2).astype(jnp.int32)
    vet_mult = 100 + 25 * rank
    dmg = jnp.where(can_fire, U_DMG[utype] * mult // 100 * vet_mult // 100, 0)
    has_splash = U_SPLASH_FP[utype] > 0
    # single-target units: direct hit to their unit target
    direct_u = jnp.where(tgt_is_b | has_splash, 0, dmg)
    dmg_u = jnp.zeros(MAX_U, jnp.int32).at[ui].add(direct_u)
    # AoE units: blast every enemy unit within splash of the target point (this
    # is how a tank/artillery shot near a harvester destroys it)
    splash2 = _r2(U_SPLASH_FP[utype])
    d2s = _d2(tx[:, None], ty[:, None], state.u_x[None, :], state.u_y[None, :])
    salive = (state.u_type >= 0) & (state.u_hp > 0)
    splash_hit = (can_fire[:, None] & has_splash[:, None] & (d2s <= splash2[:, None])
                  & (state.u_owner[:, None] != state.u_owner[None, :]) & salive[None, :])
    dmg_u = dmg_u + jnp.sum(jnp.where(splash_hit, dmg[:, None], 0), axis=0)
    dmg_b = jnp.zeros(MAX_B, jnp.int32).at[jnp.where(tgt_is_b, bi, 0)].add(
        jnp.where(tgt_is_b, dmg, 0))
    cd = jnp.where(can_fire, U_PERIOD[utype],
                   jnp.maximum(state.u_cd - 1, 0))
    new_vet = state.u_vet + jnp.where(can_fire, dmg, 0)
    # movement: toward attack-move goal when no target in range; engage-chase in sight
    has_goal = state.u_ax >= 0
    gx = jnp.where(has_tgt, tx, jnp.where(has_goal, state.u_ax, state.u_x))
    gy = jnp.where(has_tgt, ty, jnp.where(has_goal, state.u_ay, state.u_y))
    move = fighter & ~in_range & (has_tgt | has_goal)
    nx, ny = _step_toward(state.u_x, state.u_y, gx, gy, U_SPEED[utype])
    # only fighters manage their goal here — a harvester's u_ax is owned by
    # _tick_harvesters (which needs the arrival to trigger its HOLD state)
    arrived = fighter & has_goal & (_d2(state.u_x, state.u_y, state.u_ax, state.u_ay)
                                    <= _r2(jnp.int32(FP)))
    # drop a fighter's forced target once it dies (revert to auto-acquire); leave
    # harvester targets untouched — those are managed in _tick_harvesters.
    new_tgt = jnp.where(fighter, jnp.where(ft_valid, ft, -1), state.u_tgt)
    return state._replace(
        u_hp=jnp.maximum(state.u_hp - dmg_u, 0),
        b_hp=jnp.maximum(state.b_hp - dmg_b, 0),
        u_cd=cd, u_vet=new_vet, u_tgt=new_tgt,
        u_x=jnp.where(move, nx, state.u_x),
        u_y=jnp.where(move, ny, state.u_y),
        u_ax=jnp.where(arrived, -1, state.u_ax),
        u_ay=jnp.where(arrived, -1, state.u_ay))


def _resolve_collisions(state: State) -> State:
    """Units occupy space (Boom-style): overlapping same-layer units push apart
    along their separation vector, weighted by mass — light infantry yield to
    tanks, a stack of 4 spreads into a visible cluster instead of one point."""
    alive = (state.u_type >= 0) & (state.u_hp > 0)
    ut = jnp.maximum(state.u_type, 0)
    r = U_RADIUS_FP[ut]
    m = U_MASS[ut]
    idx = jnp.arange(MAX_U)
    dx = state.u_x[:, None] - state.u_x[None, :]
    dy = state.u_y[:, None] - state.u_y[None, :]
    dist = _isqrt(_d2(state.u_x[:, None], state.u_y[:, None],
                      state.u_x[None, :], state.u_y[None, :])) * 4
    rsum = r[:, None] + r[None, :]
    overlap = jnp.maximum(rsum - dist, 0)
    air = U_AIR[ut]
    pair = (alive[:, None] & alive[None, :] & (idx[:, None] != idx[None, :])
            & (air[:, None] == air[None, :]) & (overlap > 0))
    # i takes a share of separation proportional to j's mass; half per tick so
    # a stack resolves into a visible cluster within a few frames
    mag = overlap * m[None, :] // ((m[:, None] + m[None, :]) * 2)
    d_safe = jnp.maximum(dist, 1)
    ux = jnp.sign(dx) * (jnp.abs(dx) * mag // d_safe)
    uy = jnp.sign(dy) * (jnp.abs(dy) * mag // d_safe)
    # coincident bodies split along x by id order (deterministic)
    coincident = pair & (dist == 0)
    ux = jnp.where(coincident, jnp.where(idx[:, None] < idx[None, :], -mag, mag), ux)
    disp_x = jnp.sum(jnp.where(pair, ux, 0), axis=1)
    disp_y = jnp.sum(jnp.where(pair, uy, 0), axis=1)
    cap = FP // 3
    disp_x = jnp.clip(disp_x, -cap, cap)
    disp_y = jnp.clip(disp_y, -cap, cap)
    nx = jnp.clip(state.u_x + jnp.where(alive, disp_x, 0), 0, W_FP - 1)
    ny = jnp.clip(state.u_y + jnp.where(alive, disp_y, 0), 0, H_FP - 1)
    return state._replace(u_x=nx, u_y=ny)


def _building_attacks(state: State):
    """Defense buildings (turret/tower) auto-fire at the nearest enemy unit in
    range. Returns (state, dmg_to_units). Powered + operational only."""
    bt = jnp.maximum(state.b_type, 0)
    armed = ((state.b_type >= 0) & (state.b_hp > 0) & (state.b_prog == 0)
             & (B_DMG[bt] > 0))
    # power gating: unpowered defenses go dark
    p_ok = jnp.stack([power_balance(state, 0) >= 0, power_balance(state, 1) >= 0])
    armed = armed & p_ok[jnp.clip(state.b_owner, 0, 1)]
    d2 = _d2(state.b_x[:, None], state.b_y[:, None],
             state.u_x[None, :], state.u_y[None, :])
    ualive = (state.u_type >= 0) & (state.u_hp > 0)
    vic_air = U_AIR[jnp.maximum(state.u_type, 0)]
    b_aa = B_ANTIAIR[bt]
    elig = (armed[:, None] & ualive[None, :]
            & (state.b_owner[:, None] != state.u_owner[None, :])
            & ((vic_air[None, :] == 0) | (b_aa[:, None] == 1))
            & (d2 <= _r2(B_RANGE_FP[bt])[:, None]))
    BIG = jnp.int32(2**30)
    cand = jnp.where(elig, d2, BIG)
    tgt = jnp.argmin(cand, axis=1)
    fires = (cand[jnp.arange(MAX_B), tgt] < BIG) & (state.b_cd == 0)
    dmg_u = jnp.zeros(MAX_U, jnp.int32).at[tgt].add(jnp.where(fires, B_DMG[bt], 0))
    b_cd = jnp.where(fires, B_PERIOD[bt], jnp.maximum(state.b_cd - 1, 0))
    return state._replace(b_cd=b_cd), dmg_u


def _crush(state: State) -> State:
    """RA2 crush: vehicles (harvester/tank/bulldog) kill enemy INFANTRY
    (rifleman/rocketeer) they drive over. Anti-armor infantry must kill the
    vehicle from range first — that is the counter."""
    alive = (state.u_type >= 0) & (state.u_hp > 0)
    ut = jnp.maximum(state.u_type, 0)
    is_veh = alive & ((ut == 0) | (ut == 3) | (ut == 4))
    is_inf = alive & ((ut == 1) | (ut == 2)) & (U_AIR[ut] == 0)
    # infantry j crushed if any enemy vehicle i overlaps its body
    dist2 = _d2(state.u_x[:, None], state.u_y[:, None],
                state.u_x[None, :], state.u_y[None, :])
    crush_r = U_RADIUS_FP[ut][:, None] + jnp.int32(int(0.2 * FP))
    over = (is_veh[:, None] & is_inf[None, :]
            & (state.u_owner[:, None] != state.u_owner[None, :])
            & (dist2 <= _r2(crush_r)))
    crushed = jnp.any(over, axis=0)
    return state._replace(u_hp=jnp.where(crushed, 0, state.u_hp))


# ---------------------------------------------------------------- step / result
def step(state: State, actions: jax.Array) -> State:
    """actions: (2, 3) single command per player, or (2, K, 3) up to K commands
    per player per tick (selection orders need several at once)."""
    if actions.ndim == 2:
        actions = actions[:, None, :]
    for k in range(actions.shape[1]):
        state = _process_action(state, 0, actions[0, k])
        state = _process_action(state, 1, actions[1, k])
    state = state._replace(
        b_prog=jnp.maximum(state.b_prog - 1, 0),
        pend_t=jnp.where(state.pend_type >= 0, jnp.maximum(state.pend_t - 1, 0),
                         state.pend_t))
    state = _tick_production(state)
    state = _tick_harvesters(state)
    state = _tick_combat(state)
    state, def_dmg = _building_attacks(state)
    state = state._replace(u_hp=jnp.maximum(
        state.u_hp - jnp.where(state.u_hp > 0, def_dmg, 0), 0))
    state = _crush(state)
    state = _resolve_collisions(state)
    # dead units leave the pool
    gone = (state.u_hp <= 0) & (state.u_type >= 0)
    return state._replace(
        u_type=jnp.where(gone, -1, state.u_type),
        tick=state.tick + 1)


def result(state: State) -> jax.Array:
    """-1 running, 0 p0 wins, 1 p1 wins, 2 draw."""
    con0 = jnp.any((state.b_type == 0) & (state.b_owner == 0) & (state.b_hp > 0))
    con1 = jnp.any((state.b_type == 0) & (state.b_owner == 1) & (state.b_hp > 0))
    val = jnp.array([
        state.credits[0] + jnp.sum(jnp.where((state.u_owner == 0) & (state.u_hp > 0),
                                             U_COST[jnp.maximum(state.u_type, 0)], 0)),
        state.credits[1] + jnp.sum(jnp.where((state.u_owner == 1) & (state.u_hp > 0),
                                             U_COST[jnp.maximum(state.u_type, 0)], 0))])
    timeout = state.tick >= TIME_LIMIT
    by_value = jnp.where(val[0] > val[1], 0, jnp.where(val[1] > val[0], 1, 2))
    return jnp.where(~con1, 0, jnp.where(~con0, 1,
                     jnp.where(timeout, by_value, -1)))
