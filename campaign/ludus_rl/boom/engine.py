"""Boom game core: pure-functional JAX engine.

Contract (docs/01, AGENTS.md §1):
- reset(key, decks) -> State ; step(state, actions, key) -> State
- observe(state, player) -> Obs ; legal(state, player) -> mask ; result(state) -> int
- Game STATE is int32 fixed-point (positions in 1/256 tile). Floats only in observations.
- No wall clock, no global RNG. step() is deterministic given (state, actions); the key
  parameter is accepted for forward-compat and is currently unused (v1 has no stochastic
  branches inside step — the only randomness is the deck shuffle in reset).
- Illegal actions never raise and never auto-correct: they become no-ops COUNTED in
  state.illegal (AGENTS.md §2 — no silent fallbacks).

Coordinates: engine frame is player 0's frame (player 0 owns low y). Actions and
observations are PLAYER-CENTRIC: player 1's (x, y) and observation planes are rotated
180 degrees by the engine, so both players "see" themselves at the bottom. This makes
policies weight-shareable across seats.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .cards import (
    AURA_HEAL,
    AURA_RAGE,
    CARDS,
    DECK_A,
    DECK_B,
    EFFECT_PULL,
    EFFECT_RAGE,
    EFFECT_SLOW,
    EFFECT_STUN,
    FP,
)

# ---------------------------------------------------------------- constants
W, H = 18, 32                     # board tiles (x, y)
W_FP, H_FP = W * FP, H * FP
MAX_UNITS = 64
N_TOWERS = 6
N_CARDS = 96   # OBS one-hot CAPACITY (padded): card table may grow to 96 ids
# without another obs change. v11 bumped 60->96 via scripts/migrate_obs.py
# (zero-pad weight surgery, exact-equivalence preserving) across the whole
# checkpoint fleet. Table size (len(CARDS)) is independent and asserted <= this.
HAND = 4
DECK = 8

TICKS_REG = 900                   # 3 min at 5 ticks/s
TICKS_MAX = 1200                  # +60 s overtime
DOUBLE_TICK = 600                 # last 60 s of regulation: double energy

E_UNIT = 14                       # energy fixed-point: 1 energy = 14 units (2.8 s = 14 ticks)
E_MAX = 10 * E_UNIT
E_START = 5 * E_UNIT

DEPLOY_TICKS = 5                  # 1 s deploy delay
SIGHT_FP = int(5.5 * FP)          # minimum target acquisition radius
RIVER_LO_FP = 15 * FP             # river occupies tile rows 15..16
RIVER_HI_FP = 17 * FP
BRIDGE_X_FP = (int(3.5 * FP), int(14.5 * FP))   # bridge center x, lanes 0/1 (CR layout)
LANE_SPLIT_FP = 9 * FP

# spell damage to crown towers is reduced PER CARD (C.tower_pct, from source data)

# towers: index 0..2 = player 0 (turret L, turret R, core), 3..5 = player 1
# EXACT CR geometry: princess towers at (3.5, 6.5)/(14.5, 6.5) (mirrors 25.5),
# king centered x=9, y=3. The crossed-side rule (below) is what gates engagement,
# so towers never touch bridge/river units or anything across the water — and
# attackers like a musketeer (reach 7.5 c-c) can NEVER outrange a tower from
# their own side (own rows are >= 10.6 tiles from the enemy princess).
TOWER_OWNER = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
TOWER_IS_CORE = np.array([0, 0, 1, 0, 0, 1], dtype=np.int32)
TOWER_X = np.array([int(3.5 * FP), int(14.5 * FP), 9 * FP,
                    int(3.5 * FP), int(14.5 * FP), 9 * FP], dtype=np.int32)
TOWER_Y = np.array([int(6.5 * FP), int(6.5 * FP), 3 * FP,
                    int(25.5 * FP), int(25.5 * FP), 29 * FP], dtype=np.int32)
# CR tournament-standard (level 11): princess 3052hp/109dmg/0.8s, king 4824hp/109dmg/1s
TOWER_MAX_HP = np.array([3052, 3052, 4824, 3052, 3052, 4824], dtype=np.int32)
TOWER_DMG = np.full(6, 109, dtype=np.int32)
TOWER_PERIOD_ARR = np.where(TOWER_IS_CORE == 1, 5, 4).astype(np.int32)
# center-to-center reach = CR edge range (7.5 / king 7) + tower radius + unit radius
# CR tournament values: princess tower 7.5 tiles, king tower 7.0 — the king
# engaging from 9.5 made it fire far too early (user-reported v9 fix)
TOWER_AIM_TICKS = 2      # ~0.4s first-shot aim delay on a fresh target (CR-style)
TOWER_RANGE_ARR = np.where(TOWER_IS_CORE == 1, int(7.0 * FP), int(7.5 * FP)).astype(np.int32)
# ground units cannot enter the river except on the bridge spans
BRIDGE_SPANS_FP = ((3 * FP, 5 * FP), (14 * FP, 16 * FP))
# target-radius padding (fp): attacking a fat tower counts its footprint
UNIT_PAD_FP = FP // 2
TOWER_PAD = np.where(TOWER_IS_CORE == 1, 2 * FP, int(1.5 * FP)).astype(np.int32)

# spawn formation offsets (fp), used in id order for multi-body cards (up to 15)
SPAWN_OFF = np.array(
    [(0, 0), (128, 0), (-128, 0), (0, 128), (0, -128), (128, 128), (-128, -128),
     (128, -128), (-128, 128), (256, 0), (-256, 0), (0, 256), (0, -256),
     (256, 128), (-256, -128), (256, -128)],
    dtype=np.int32,
)

# status word bit fields (stun uses 7 bits to keep the int32 sign bit clear)
_SLOW_SHIFT, _RAGE_SHIFT, _DELAY_SHIFT, _STUN_SHIFT = 0, 8, 16, 24
_BYTE = 0xFF
_STUN_MASK = 0x7F

from .cards import ARCHETYPES as _ARCH
BUILDING_ARCH = _ARCH.index("building")
CHARGE_FP = int(3.5 * 256)   # prince-family run-up distance to become charged

OBS_C = 28          # 12 archetypes x 2 owners + 2 hp planes + 2 tower planes
OBS_VEC = 4 + N_TOWERS + 2 * HAND * N_CARDS   # 490

RESULT_ONGOING, RESULT_P0, RESULT_P1, RESULT_DRAW = -1, 0, 1, 2

# card table as device constants
C = jax.tree_util.tree_map(jnp.asarray, CARDS)


# ---------------------------------------------------------------- state
class State(NamedTuple):
    u_owner: jax.Array   # (64,) i32: 0/1 (meaningless when dead)
    u_type: jax.Array    # (64,) i32: card id
    u_hp: jax.Array      # (64,) i32: alive iff > 0
    u_x: jax.Array       # (64,) i32 fp
    u_y: jax.Array       # (64,) i32 fp
    u_tgt: jax.Array     # (64,) i32: -1 none, 0..63 unit, 64..69 tower (STICKY, CR-style)
    u_cd: jax.Array      # (64,) i32: attack cooldown ticks
    u_charge: jax.Array  # (64,) i32: charge run-up distance (fp); >=CHARGE_FP => charged
    u_status: jax.Array  # (64,) i32: slow | rage<<8 | deploy_delay<<16 | stun<<24
    tower_hp: jax.Array  # (6,) i32
    tower_cd: jax.Array  # (6,) i32
    tower_tgt: jax.Array  # (6,) i32: sticky tower target, -1 none
    tower_stun: jax.Array  # (6,) i32: freeze/zap ticks remaining (tower cannot fire)
    spells: jax.Array    # (8, 5) i32 pending spells: card,owner,x_fp,y_fp,ticks_left
                         # (card == -1 => free slot); lobbed spells impact on arrival
    energy: jax.Array    # (2,) i32, units of 1/14 energy
    hand: jax.Array      # (2,4) i32 card ids
    queue: jax.Array     # (2,4) i32 card ids
    tick: jax.Array      # () i32
    key: jax.Array       # PRNG key from reset (provenance)
    illegal: jax.Array   # (2,) i32: counted illegal-action no-ops


class Obs(NamedTuple):
    spatial: jax.Array   # (H, W, OBS_C) f32, player-centric
    vector: jax.Array    # (OBS_VEC,) f32


def _slow(s):
    return (s >> _SLOW_SHIFT) & _BYTE


def _rage(s):
    return (s >> _RAGE_SHIFT) & _BYTE


def _delay(s):
    return (s >> _DELAY_SHIFT) & _BYTE


def _stun(s):
    return (s >> _STUN_SHIFT) & _STUN_MASK


def _pack(slow, rage, delay, stun=0):
    return ((slow << _SLOW_SHIFT) | (rage << _RAGE_SHIFT)
            | (delay << _DELAY_SHIFT) | (stun << _STUN_SHIFT))


# ---------------------------------------------------------------- geometry helpers
def _dist2_16(ax, ay, bx, by):
    """Squared distance in (fp/16) units — fits int32 across the whole board.

    Uses |d|//16 (truncation toward zero), NOT d//16: floor division rounds toward
    -inf, which would make distances measured in the -y/-x direction read larger —
    a systematic range/speed advantage for player 1 (found via mirror-deck control:
    53% vs 28% seat bias). All engine arithmetic must be mirror-symmetric."""
    dx = jnp.abs(ax - bx) // 16
    dy = jnp.abs(ay - by) // 16
    return dx * dx + dy * dy


def _r2_16(r_fp):
    r = r_fp // 16
    return r * r


def _isqrt(d2):
    """Exact integer sqrt for 0 <= d2 < 2^24, deterministic on CPU and GPU.

    float32 represents these ints exactly and IEEE sqrt is correctly rounded,
    so a +-1 correction makes the result exact."""
    s = jnp.sqrt(d2.astype(jnp.float32)).astype(jnp.int32)
    s = jnp.where(s * s > d2, s - 1, s)
    s = jnp.where((s + 1) * (s + 1) <= d2, s + 1, s)
    return s


def _flip_xy(x, y):
    """180-degree board rotation mapping tile centers to tile centers."""
    return W_FP - x, H_FP - y


# ---------------------------------------------------------------- reset
def reset(key: jax.Array, decks: jax.Array | None = None) -> State:
    """decks: (2, 8) int32 card ids. Hands are a shuffled draw of the 8-card cycle."""
    if decks is None:
        decks = jnp.stack([jnp.asarray(DECK_A), jnp.asarray(DECK_B)])
    k0, k1 = jax.random.fold_in(key, 0), jax.random.fold_in(key, 1)
    d0 = jax.random.permutation(k0, decks[0])
    d1 = jax.random.permutation(k1, decks[1])
    shuf = jnp.stack([d0, d1]).astype(jnp.int32)
    z64 = jnp.zeros(MAX_UNITS, jnp.int32)
    return State(
        u_owner=z64, u_type=z64, u_hp=z64, u_x=z64, u_y=z64,
        u_tgt=jnp.full(MAX_UNITS, -1, jnp.int32), u_cd=z64, u_status=z64,
        u_charge=z64,
        tower_hp=jnp.asarray(TOWER_MAX_HP), tower_cd=jnp.zeros(N_TOWERS, jnp.int32),
        tower_tgt=jnp.full(N_TOWERS, -1, jnp.int32),
        tower_stun=jnp.zeros(N_TOWERS, jnp.int32),
        spells=jnp.full((8, 5), -1, jnp.int32),
        energy=jnp.full(2, E_START, jnp.int32),
        hand=shuf[:, :HAND], queue=shuf[:, HAND:],
        tick=jnp.int32(0), key=key, illegal=jnp.zeros(2, jnp.int32),
    )


# ---------------------------------------------------------------- card play
def _capacity_ok(u_hp, card):
    free = jnp.sum(u_hp <= 0)
    return free >= C.count[card]


def _own_half_ok(p, y_tile):
    return jnp.where(p == 0, y_tile <= 14, y_tile >= 17)


def _pocket_ok(state: State, p, x_tile, y_tile):
    """CR rule: destroying an enemy princess tower unlocks that lane's pocket —
    the opponent-half rectangle from the river up to the tower row (rows 17..25
    in the attacker's direction), on that tower's half of the board."""
    lane = (x_tile >= 9).astype(jnp.int32) if hasattr(x_tile, "astype") else jnp.int32(x_tile >= 9)
    enemy_turret_dead = jnp.where(
        p == 0,
        jnp.where(lane == 0, state.tower_hp[3] == 0, state.tower_hp[4] == 0),
        jnp.where(lane == 0, state.tower_hp[0] == 0, state.tower_hp[1] == 0))
    in_pocket_rows = jnp.where(p == 0,
                               (y_tile >= 17) & (y_tile <= 23),
                               (y_tile >= 8) & (y_tile <= 14))
    return enemy_turret_dead & in_pocket_rows


def _play_legal(state: State, p, slot, x_tile, y_tile):
    """Legality of playing hand[slot] at engine-frame tile (x, y)."""
    card = state.hand[p, slot]
    on_board = (x_tile >= 0) & (x_tile < W) & (y_tile >= 0) & (y_tile < H)
    afford = state.energy[p] >= C.cost[card] * E_UNIT
    # spells: any tile; anywhere-deployers (miner-like): any tile but need pool room;
    # normal units: (own half OR unlocked tower pocket) + pool room
    ground_ok = _own_half_ok(p, y_tile) | _pocket_ok(state, p, x_tile, y_tile)
    place_ok = jnp.where(
        C.is_spell[card] == 1, True,
        jnp.where(C.anywhere[card] == 1, _capacity_ok(state.u_hp, card),
                  ground_ok & _capacity_ok(state.u_hp, card)))
    return on_board & afford & place_ok


def _apply_spell(state: State, p, card, cx, cy, do):
    """Spell IMPACT at fp center (cx, cy). All writes gated by `do`."""
    r2 = _r2_16(C.splash_fp[card])
    alive = state.u_hp > 0
    in_r = _dist2_16(state.u_x, state.u_y, cx, cy) <= r2
    can_hit_air = (C.targets_air[card] == 1) | (C.air[state.u_type] == 0)
    enemy = alive & (state.u_owner != p) & in_r & can_hit_air
    ally = alive & (state.u_owner == p) & in_r

    dmg = C.dmg[card]
    hp = jnp.where(do & enemy & (dmg > 0), jnp.maximum(state.u_hp - dmg, 0), state.u_hp)

    slow, rage = _slow(state.u_status), _rage(state.u_status)
    delay, stun = _delay(state.u_status), _stun(state.u_status)
    dur = C.duration[card]
    slow = jnp.where(do & enemy & (C.effect[card] == EFFECT_SLOW), jnp.maximum(slow, dur), slow)
    rage = jnp.where(do & ally & (C.effect[card] == EFFECT_RAGE), jnp.maximum(rage, dur), rage)
    # stun (zap/freeze family): halts the victim AND resets its target lock (CR rule)
    stunned = do & enemy & (C.effect[card] == EFFECT_STUN)
    stun = jnp.where(stunned, jnp.maximum(stun, dur), stun)
    tgt = jnp.where(stunned, -1, state.u_tgt)

    t_in = _dist2_16(jnp.asarray(TOWER_X), jnp.asarray(TOWER_Y), cx, cy) <= r2
    t_enemy = (jnp.asarray(TOWER_OWNER) != p) & (state.tower_hp > 0) & t_in
    t_dmg = dmg * C.tower_pct[card] // 100        # per-card crown-tower reduction
    tower_hp = jnp.where(do & t_enemy & (dmg > 0),
                         jnp.maximum(state.tower_hp - t_dmg, 0), state.tower_hp)
    # zap/freeze family stuns towers too (CR: a frozen tower stops firing)
    t_stun = jnp.where(do & t_enemy & (C.effect[card] == EFFECT_STUN),
                       jnp.maximum(state.tower_stun, dur), state.tower_stun)

    # knockback (fireball/snowball family): push surviving enemy units away from
    # the blast center along the center->victim direction (sign-symmetric math)
    kb = C.kb_fp[card]
    # CR: buildings are anchored and HEAVY units (giant/golem/pekka class,
    # mass >= 10) resist pushback entirely (v13 user-reported)
    kb_immune = ((C.archetype[state.u_type] == BUILDING_ARCH)
                 | (C.mass[state.u_type] >= 10))
    pushed = do & enemy & (kb > 0) & (hp > 0) & ~kb_immune \
        & (C.effect[card] != EFFECT_PULL)
    dx, dy = state.u_x - cx, state.u_y - cy
    dist = _isqrt(_dist2_16(state.u_x, state.u_y, cx, cy)) * 16
    ux = jnp.sign(dx) * (jnp.abs(dx) * kb // jnp.maximum(dist, 1))
    uy = jnp.sign(dy) * (jnp.abs(dy) * kb // jnp.maximum(dist, 1))
    # dead-center hits push straight up-board away from the caster
    ux = jnp.where(dist == 0, 0, ux)
    uy = jnp.where(dist == 0, jnp.where(p == 0, kb, -kb), uy)
    nx = jnp.clip(jnp.where(pushed, state.u_x + ux, state.u_x), 0, W_FP - 1)
    ny = jnp.clip(jnp.where(pushed, state.u_y + uy, state.u_y), 0, H_FP - 1)
    # v15 Whirlgale (tornado family): PULL toward the blast center. CR: drags
    # everything including heavies — only buildings are anchored. Capped so a
    # victim never overshoots the center.
    is_pull = C.effect[card] == EFFECT_PULL
    pull_immune = C.archetype[state.u_type] == BUILDING_ARCH
    pulled = do & enemy & is_pull & (hp > 0) & ~pull_immune
    pmag = jnp.minimum(kb, dist)                    # stop AT the center
    pux = -jnp.sign(dx) * (jnp.abs(dx) * pmag // jnp.maximum(dist, 1))
    puy = -jnp.sign(dy) * (jnp.abs(dy) * pmag // jnp.maximum(dist, 1))
    nx = jnp.clip(jnp.where(pulled, nx + pux, nx), 0, W_FP - 1)
    ny = jnp.clip(jnp.where(pulled, ny + puy, ny), 0, H_FP - 1)
    # v13 user-reported: knockback could dump ground units INTO the river where
    # they were unreachable and stuck — same water rule as collisions applies
    nx, ny = _river_guard(state, nx, ny)
    state = state._replace(u_hp=hp, u_status=_pack(slow, rage, delay, stun),
                           u_tgt=tgt, u_x=nx, u_y=ny, tower_hp=tower_hp,
                           tower_stun=t_stun)
    # v16 barrel family: a spell can SPAWN units (goblin/barbarian barrels).
    # On impact, drop C.count[card] bodies of spawn_type at the target for the
    # caster, into the first free slots (same allocation as a played unit card).
    st = C.spawn_type[card]
    def _spawn_from_spell(state):
        free = state.u_hp <= 0
        rank = jnp.cumsum(free) - 1
        body = jnp.arange(SPAWN_OFF.shape[0])
        want = body < C.count[card]
        place = free[None, :] & (rank[None, :] == body[:, None]) & want[:, None]
        sel = place.any(axis=0)
        which = jnp.argmax(place, axis=0)
        off = jnp.asarray(SPAWN_OFF)
        sx = jnp.clip(cx + off[which, 0], 0, W_FP - 1)
        sy = jnp.clip(cy + off[which, 1], 0, H_FP - 1)
        return state._replace(
            u_owner=jnp.where(sel, p, state.u_owner),
            u_type=jnp.where(sel, st, state.u_type),
            u_hp=jnp.where(sel, C.hp[st], state.u_hp),
            u_x=jnp.where(sel, sx, state.u_x),
            u_y=jnp.where(sel, sy, state.u_y),
            u_tgt=jnp.where(sel, -1, state.u_tgt),
            u_cd=jnp.where(sel, 0, state.u_cd),
            u_charge=jnp.where(sel, 0, state.u_charge),
            u_status=jnp.where(sel, _pack(0, 0, DEPLOY_TICKS), state.u_status))
    return jax.lax.cond(do & (st >= 0), _spawn_from_spell, lambda s: s, state)


def _apply_spawn(state: State, p, card, cx, cy, do):
    """Spawn C.count[card] bodies around (cx, cy) into first-free slots (id order)."""
    free = state.u_hp <= 0
    rank = jnp.cumsum(free) - 1                                   # rank among free slots
    body = jnp.arange(SPAWN_OFF.shape[0])                          # (8,)
    want = body < C.count[card]                                    # (8,)
    place = do & free[None, :] & (rank[None, :] == body[:, None]) & want[:, None]  # (8, 64)
    sel = place.any(axis=0)                                        # (64,) slot receives a body
    which = jnp.argmax(place, axis=0)                              # body index per slot

    off = jnp.asarray(SPAWN_OFF)
    nx = jnp.clip(cx + off[which, 0], 0, W_FP - 1)
    ny = jnp.clip(cy + off[which, 1], 0, H_FP - 1)
    return state._replace(
        u_owner=jnp.where(sel, p, state.u_owner),
        u_type=jnp.where(sel, card, state.u_type),
        u_hp=jnp.where(sel, C.hp[card], state.u_hp),
        u_x=jnp.where(sel, nx, state.u_x),
        u_y=jnp.where(sel, ny, state.u_y),
        u_tgt=jnp.where(sel, -1, state.u_tgt),
        u_cd=jnp.where(sel, 0, state.u_cd),
        u_status=jnp.where(sel, _pack(0, 0, DEPLOY_TICKS), state.u_status),
    )


def _cycle_hand(state: State, p, slot, do):
    played = state.hand[p, slot]
    new_hand_p = state.hand[p].at[slot].set(state.queue[p, 0])
    new_queue_p = jnp.concatenate([state.queue[p, 1:], played[None]])
    hand = jnp.where(do, state.hand.at[p].set(new_hand_p), state.hand)
    queue = jnp.where(do, state.queue.at[p].set(new_queue_p), state.queue)
    return state._replace(hand=hand, queue=queue)


def _process_action(state: State, p, action):
    """One player's action (already engine-frame). Sequential: p0 then p1 (deterministic
    slot allocation; the asymmetry is at most one tick of spawn priority)."""
    slot, x_tile, y_tile = action[0], action[1], action[2]
    is_play = slot < HAND
    slot_c = jnp.clip(slot, 0, HAND - 1)
    ok = _play_legal(state, p, slot_c, x_tile, y_tile)
    do = is_play & ok
    card = state.hand[p, slot_c]
    cx = jnp.clip(x_tile, 0, W - 1) * FP + FP // 2
    cy = jnp.clip(y_tile, 0, H - 1) * FP + FP // 2

    state = state._replace(
        energy=jnp.where(do, state.energy.at[p].add(-C.cost[card] * E_UNIT), state.energy),
        illegal=jnp.where(is_play & ~ok, state.illegal.at[p].add(1), state.illegal),
    )
    spell = C.is_spell[card] == 1
    delay = C.spell_delay[card]
    # instant spells apply now; lobbed spells enqueue and impact on arrival
    state = _apply_spell(state, p, card, cx, cy, do & spell & (delay == 0))
    free = state.spells[:, 0] < 0
    slot = jnp.argmax(free)
    row = jnp.stack([card, jnp.int32(p), cx, cy, delay])
    enq = do & spell & (delay > 0) & free.any()
    spells = jnp.where(enq, state.spells.at[slot].set(row), state.spells)
    state = state._replace(spells=spells)
    state = _apply_spawn(state, p, card, cx, cy, do & ~spell)
    return _cycle_hand(state, p, slot_c, do)


# ---------------------------------------------------------------- per-tick systems
def _tick_status(state: State) -> State:
    s = state.u_status
    slow = jnp.maximum(_slow(s) - 1, 0)
    rage = jnp.maximum(_rage(s) - 1, 0)
    delay = jnp.maximum(_delay(s) - 1, 0)
    stun = jnp.maximum(_stun(s) - 1, 0)
    return state._replace(u_status=_pack(slow, rage, delay, stun))


def _apply_auras(state: State) -> State:
    alive = state.u_hp > 0
    atype = C.aura_type[state.u_type]
    src = alive & (atype != 0)                                     # (64,) aura sources
    r2 = _r2_16(C.aura_radius_fp[state.u_type])                    # per source
    d2 = _dist2_16(state.u_x[:, None], state.u_y[:, None],
                   state.u_x[None, :], state.u_y[None, :])         # (src, dst)
    ally = src[:, None] & alive[None, :] & (state.u_owner[:, None] == state.u_owner[None, :])
    in_r = d2 <= r2[:, None]

    raged = (ally & in_r & (atype[:, None] == AURA_RAGE)).any(axis=0)
    slow, rage = _slow(state.u_status), _rage(state.u_status)
    delay, stun = _delay(state.u_status), _stun(state.u_status)
    rage = jnp.where(raged, jnp.maximum(rage, 2), rage)

    heal = jnp.sum(jnp.where(ally & in_r & (atype[:, None] == AURA_HEAL),
                             C.aura_power[state.u_type][:, None], 0), axis=0)
    max_hp = C.hp[state.u_type]
    hp = jnp.where(alive, jnp.minimum(state.u_hp + heal, max_hp), state.u_hp)
    return state._replace(u_hp=hp, u_status=_pack(slow, rage, delay, stun))


def _acquire_targets(state: State) -> State:
    """CR-style STICKY targeting: a unit keeps its locked target until the target
    dies (or the unit is stunned — the stun handler resets locks). Only lock-less
    units acquire: nearest eligible enemy within sight, ties broken by lower id."""
    alive = state.u_hp > 0
    utype = state.u_type
    can_target = (alive & (C.dmg[utype] > 0) & (_delay(state.u_status) == 0)
                  & (_stun(state.u_status) == 0))

    # --- validate existing locks (sticky unless target is gone) ---
    cur = state.u_tgt
    cur_is_tower = cur >= MAX_UNITS
    ui = jnp.clip(cur, 0, MAX_UNITS - 1)
    ti = jnp.clip(cur - MAX_UNITS, 0, N_TOWERS - 1)
    unit_ok = (alive[ui] & (state.u_owner[ui] != state.u_owner)
               & ((C.targets_air[utype] == 1) | (C.air[state.u_type[ui]] == 0)))
    tower_ok = (state.tower_hp[ti] > 0) \
        & (jnp.asarray(TOWER_OWNER)[ti] != state.u_owner)
    # CR: building-targeters do NOT hold locks — they continuously go for the
    # nearest enemy STRUCTURE, so a freshly deployed cannon pulls an already
    # marching giant (structures are static, so this is oscillation-free)
    #
    # v14 (user-reported vs CR): normal troops hold a lock only while ENGAGED
    # (target inside attack reach). A troop still WALKING to its target
    # re-evaluates — placing an ice-golem-class body between a mega minion and
    # its victim pulls the minion (CR kiting). Engaged uses the exact reach
    # formula of _unit_attacks so lock semantics match hit semantics.
    txp, typ = _tgt_pos(state)
    d2_cur = _dist2_16(state.u_x, state.u_y, txp, typ)
    engaged = (cur >= 0) & (d2_cur <= _r2_16(C.range_fp[utype] + _tgt_pad(state)))
    keep = (cur >= 0) & can_target & (C.bldg_only[utype] == 0) & engaged \
        & jnp.where(cur_is_tower, tower_ok, unit_ok)

    # --- acquisition for lock-less units ---
    # CR building-targeter rule: bldg_only units consider enemy BUILDINGS (both
    # deployed building-units and towers) and know about them GLOBALLY — placing a
    # cannon re-routes a hog from anywhere. Everyone else: units within sight.
    is_bldg_unit = C.archetype[utype] == BUILDING_ARCH
    bo = C.bldg_only[utype] == 1

    # unit-vs-unit eligibility (i = attacker, j = victim)
    d2_u = _dist2_16(state.u_x[:, None], state.u_y[:, None],
                     state.u_x[None, :], state.u_y[None, :])
    elig_u = (can_target[:, None] & alive[None, :]
              & (state.u_owner[:, None] != state.u_owner[None, :])
              & ((C.targets_air[utype][:, None] == 1) | (C.air[utype][None, :] == 0))
              & (~bo[:, None] | is_bldg_unit[None, :]))

    # unit-vs-tower (CR: defensive buildings never target towers)
    d2_t = _dist2_16(state.u_x[:, None], state.u_y[:, None],
                     jnp.asarray(TOWER_X)[None, :], jnp.asarray(TOWER_Y)[None, :])
    elig_t = (can_target[:, None] & (state.tower_hp > 0)[None, :]
              & (C.no_tower[utype] == 0)[:, None]
              & (state.u_owner[:, None] != jnp.asarray(TOWER_OWNER)[None, :]))

    # sight: at least SIGHT_FP, never less than attack reach; building-targeters
    # see structures globally
    sight_fp = jnp.maximum(jnp.int32(SIGHT_FP), C.range_fp[utype] + 2 * FP)
    sight2 = _r2_16(sight_fp)[:, None]
    BIG = jnp.int32(2**30)
    unlimited = jnp.int32(2**29)
    sight2_u = jnp.where(bo[:, None] & is_bldg_unit[None, :], unlimited, sight2)
    sight2_t = jnp.where(bo[:, None], unlimited, sight2)
    # CR pathing: building-targeters prefer princess towers over the king (the
    # king sits behind the princess line in true pathing distance) — same bias
    # as the march destination, applied only to bldg_only acquisition
    king_pref = jnp.where((jnp.asarray(TOWER_IS_CORE)[None, :] == 1) & bo[:, None],
                          _r2_16(jnp.int32(8 * FP)), 0)
    cand = jnp.concatenate([jnp.where(elig_u & (d2_u <= sight2_u), d2_u, BIG),
                            jnp.where(elig_t & (d2_t <= sight2_t),
                                      d2_t + king_pref, BIG)], axis=1)
    best = jnp.argmin(cand, axis=1).astype(jnp.int32)             # ties -> lower index
    fresh = jnp.where(cand[jnp.arange(MAX_UNITS), best] < BIG, best, -1)
    tgt = jnp.where(keep, cur, jnp.where(can_target, fresh, -1))
    return state._replace(u_tgt=tgt)


def _tgt_pos(state: State):
    """Position of each unit's target (undefined where u_tgt == -1)."""
    t = jnp.maximum(state.u_tgt, 0)
    is_tower = state.u_tgt >= MAX_UNITS
    ti = jnp.clip(t - MAX_UNITS, 0, N_TOWERS - 1)
    ui = jnp.clip(t, 0, MAX_UNITS - 1)
    tx = jnp.where(is_tower, jnp.asarray(TOWER_X)[ti], state.u_x[ui])
    ty = jnp.where(is_tower, jnp.asarray(TOWER_Y)[ti], state.u_y[ui])
    return tx, ty


def _tgt_pad(state: State):
    """Target-radius padding: CR ranges are edge-to-edge, so reaching a 3x3 turret
    or 4x4 king counts its footprint. Units pad by half a tile."""
    is_tower = state.u_tgt >= MAX_UNITS
    ti = jnp.clip(state.u_tgt - MAX_UNITS, 0, N_TOWERS - 1)
    return jnp.where(is_tower, jnp.asarray(TOWER_PAD)[ti], UNIT_PAD_FP)


def _river_side(y):
    """0 = player-0 side, 1 = player-1 side, 2 = on the river rows."""
    return jnp.where(y < RIVER_LO_FP, 0, jnp.where(y >= RIVER_HI_FP, 1, 2))


def _on_bridge_x(x):
    (a0, a1), (b0, b1) = BRIDGE_SPANS_FP
    return ((x >= a0) & (x < a1)) | ((x >= b0) & (x < b1))


def _river_guard(state: State, nx, ny):
    """CR terrain rule: the river is impassable for ground non-jumping units except
    on the bridge spans. Blocks (a) stepping INTO river rows off-bridge and
    (b) sliding OFF the bridge span while on the river rows."""
    ground = (C.air[state.u_type] == 0) & (C.jumps[state.u_type] == 0)
    new_in_river = _river_side(ny) == 2
    entering_off_bridge = ground & new_in_river & ~_on_bridge_x(nx)
    ny = jnp.where(entering_off_bridge, state.u_y, ny)
    still_in_river = ground & (_river_side(ny) == 2)
    nx = jnp.where(still_in_river & ~_on_bridge_x(nx), state.u_x, nx)
    return nx, ny


def _ground_waypoint(owner, x, y, nt_x, nt_y):
    """Lane-follow waypoint in canonical (player-0 attacks upward) frame via mirroring."""
    # mirror player-1 units into the canonical frame
    mx = jnp.where(owner == 1, W_FP - x, x)
    my = jnp.where(owner == 1, H_FP - y, y)
    lane = (mx >= LANE_SPLIT_FP).astype(jnp.int32)
    bx = jnp.where(lane == 0, BRIDGE_X_FP[0], BRIDGE_X_FP[1])
    pre = my < 14 * FP
    crossing = (my >= 14 * FP) & (my < RIVER_HI_FP)
    misaligned = jnp.abs(mx - bx) > FP // 2
    wx = bx  # x always heads for the lane's bridge until past the river
    wy = jnp.where(pre, 14 * FP + FP // 2,
                   jnp.where(crossing & misaligned, my, RIVER_HI_FP + FP // 2))
    # mirror the enemy-tower destination into canonical frame too
    mtx = jnp.where(owner == 1, W_FP - nt_x, nt_x)
    mty = jnp.where(owner == 1, H_FP - nt_y, nt_y)
    past = my >= RIVER_HI_FP
    wx = jnp.where(past, mtx, wx)
    wy = jnp.where(past, mty, wy)
    # mirror back
    return jnp.where(owner == 1, W_FP - wx, wx), jnp.where(owner == 1, H_FP - wy, wy)


def _path_lin16(ux, uy, tx, ty, cross):
    """Approximate CR pathing distance (linear, /16 scale): straight when on the
    target's side (or airborne), else via the cheaper bridge."""
    direct = _isqrt(_dist2_16(ux, uy, tx, ty))
    b1x, b1y = jnp.int32((BRIDGE_SPANS_FP[0][0] + BRIDGE_SPANS_FP[0][1]) // 2), jnp.int32(16 * FP)
    b2x, b2y = jnp.int32((BRIDGE_SPANS_FP[1][0] + BRIDGE_SPANS_FP[1][1]) // 2), jnp.int32(16 * FP)
    via1 = _isqrt(_dist2_16(ux, uy, b1x, b1y)) + _isqrt(_dist2_16(b1x, b1y, tx, ty))
    via2 = _isqrt(_dist2_16(ux, uy, b2x, b2y)) + _isqrt(_dist2_16(b2x, b2y, tx, ty))
    return jnp.where(cross, jnp.minimum(via1, via2), direct)


def _nearest_enemy_tower(state: State):
    """March destination: nearest enemy tower — except building-targeters, whose
    destination is the nearest enemy STRUCTURE (deployed buildings included), which
    is what makes a mid-placed cannon pull a hog off its tower line."""
    # v11: LINEAR pathing distance (via the cheaper bridge when across the
    # river for ground units) replaces euclidean — fixes lane choice when the
    # euclidean-nearest building is across the water but path-far
    flies_m = (C.air[state.u_type] == 1) | (C.jumps[state.u_type] == 1)
    cross_m = (~flies_m[:, None]) & (_river_side(state.u_y)[:, None]
                                     != _river_side(jnp.asarray(TOWER_Y))[None, :])
    d_lin = _path_lin16(state.u_x[:, None], state.u_y[:, None],
                        jnp.asarray(TOWER_X)[None, :], jnp.asarray(TOWER_Y)[None, :],
                        cross_m)
    enemy_t = (state.u_owner[:, None] != jnp.asarray(TOWER_OWNER)[None, :]) \
        & (state.tower_hp > 0)[None, :]
    d_lin = jnp.where(enemy_t, d_lin, 2**28)
    # king sits behind the princess line: 8-tile linear bias (user-verified v9)
    king_bias = jnp.where(jnp.asarray(TOWER_IS_CORE)[None, :] == 1,
                          jnp.int32(8 * 16), 0)
    d_lin = jnp.where(d_lin < 2**28, d_lin + king_bias, d_lin)
    ti = jnp.argmin(d_lin, axis=1)
    best_t = d_lin[jnp.arange(MAX_UNITS), ti]
    tx = jnp.asarray(TOWER_X)[ti]
    ty = jnp.asarray(TOWER_Y)[ti]

    bo = C.bldg_only[state.u_type] == 1
    is_bldg = (state.u_hp > 0) & (C.archetype[state.u_type] == BUILDING_ARCH)
    cross_b = (~flies_m[:, None]) & (_river_side(state.u_y)[:, None]
                                     != _river_side(state.u_y)[None, :])
    d_b = _path_lin16(state.u_x[:, None], state.u_y[:, None],
                      state.u_x[None, :], state.u_y[None, :], cross_b)
    enemy_b = is_bldg[None, :] & (state.u_owner[:, None] != state.u_owner[None, :])
    d_b = jnp.where(enemy_b, d_b, 2**28)
    bi = jnp.argmin(d_b, axis=1)
    best_b = d_b[jnp.arange(MAX_UNITS), bi]
    prefer_b = bo & (best_b < best_t)
    return (jnp.where(prefer_b, state.u_x[bi], tx),
            jnp.where(prefer_b, state.u_y[bi], ty))


def _move(state: State) -> State:
    alive = state.u_hp > 0
    utype = state.u_type
    speed = C.speed[utype]
    slow = _slow(state.u_status) > 0
    rage = _rage(state.u_status) > 0
    v = jnp.where(slow, speed // 2, speed)
    v = jnp.where(rage, v * 13 // 10, v)
    movable = alive & (_delay(state.u_status) == 0) & (v > 0) \
        & (_stun(state.u_status) == 0)

    tx, ty = _tgt_pos(state)
    has_tgt = state.u_tgt >= 0
    d2_tgt = _dist2_16(state.u_x, state.u_y, tx, ty)
    reach = C.range_fp[utype] + _tgt_pad(state)
    in_range = has_tgt & (d2_tgt <= _r2_16(reach))
    # spirit leap (CR): a short quick hop over the last stretch only — kamikazes
    # triple their speed once within one tile of attack reach
    leap = (C.suicide[utype] == 1) & has_tgt & (d2_tgt <= _r2_16(reach + FP))
    v = jnp.where(leap, v * 3, v)

    nt_x, nt_y = _nearest_enemy_tower(state)
    wx, wy = _ground_waypoint(state.u_owner, state.u_x, state.u_y, nt_x, nt_y)

    # river-jumpers (hog family) path straight like fliers but stay ground units
    flies = (C.air[utype] == 1) | (C.jumps[utype] == 1)
    same_side = _river_side(state.u_y) == _river_side(ty)
    chase = has_tgt & (flies | (same_side & (_river_side(state.u_y) != 2)))

    dest_x = jnp.where(chase, tx, jnp.where(flies, nt_x, wx))
    dest_y = jnp.where(chase, ty, jnp.where(flies, nt_y, wy))

    dx, dy = dest_x - state.u_x, dest_y - state.u_y
    dist = _isqrt(_dist2_16(state.u_x, state.u_y, dest_x, dest_y)) * 16   # ~fp
    do = movable & ~in_range & (dist > 0)
    arrive = dist <= v
    # sign-symmetric division: |d|*v//dist, then re-apply sign (floor division would
    # move negative directions faster — same mirror-asymmetry as _dist2_16)
    step_x = jnp.where(arrive, dx, jnp.sign(dx) * (jnp.abs(dx) * v // jnp.maximum(dist, 1)))
    step_y = jnp.where(arrive, dy, jnp.sign(dy) * (jnp.abs(dy) * v // jnp.maximum(dist, 1)))
    # charge (prince family): run-up accumulates while moving; charged units
    # move 60% faster; any stop (attack, stun, idle) resets the run-up
    chargers = C.charge[utype] == 1
    charged = chargers & (state.u_charge >= CHARGE_FP)
    v = jnp.where(charged, v * 8 // 5, v)
    step_x = jnp.where(arrive, dx, jnp.sign(dx) * (jnp.abs(dx) * v // jnp.maximum(dist, 1)))
    step_y = jnp.where(arrive, dy, jnp.sign(dy) * (jnp.abs(dy) * v // jnp.maximum(dist, 1)))
    nx = jnp.clip(state.u_x + jnp.where(do, step_x, 0), 0, W_FP - 1)
    ny = jnp.clip(state.u_y + jnp.where(do, step_y, 0), 0, H_FP - 1)
    charge = jnp.where(chargers & do & (_stun(state.u_status) == 0),
                       jnp.minimum(state.u_charge + v, 2 * CHARGE_FP),
                       jnp.where(chargers & movable & in_range, state.u_charge, 0))
    state = state._replace(u_charge=charge)
    nx, ny = _river_guard(state, nx, ny)
    return state._replace(u_x=nx, u_y=ny)


def _resolve_collisions(state: State) -> State:
    """CR-style body collision: units occupy space. Overlapping same-layer units
    (ground-ground / air-air) push apart along their separation vector, weighted by
    mass — a skeleton (mass 1) cannot budge a golem (mass 20), the golem plows
    through. Buildings (speed 0, mass ~99) are effectively immovable. Ground units
    are also pushed out of tower footprints. All arithmetic is |d|-based and
    tie-broken by id so it stays mirror-symmetric and deterministic."""
    alive = state.u_hp > 0
    utype = state.u_type
    r = C.col_r_fp[utype]
    m = C.mass[utype]
    is_air = C.air[utype] == 1
    idx = jnp.arange(MAX_UNITS)

    dx = state.u_x[:, None] - state.u_x[None, :]
    dy = state.u_y[:, None] - state.u_y[None, :]
    dist = _isqrt(_dist2_16(state.u_x[:, None], state.u_y[:, None],
                            state.u_x[None, :], state.u_y[None, :])) * 16
    rsum = r[:, None] + r[None, :]
    overlap = jnp.maximum(rsum - dist, 0)
    pair = (alive[:, None] & alive[None, :]
            & (is_air[:, None] == is_air[None, :])
            & (idx[:, None] != idx[None, :]) & (overlap > 0))
    # i's share of the separation is proportional to j's mass; resolve only a
    # QUARTER of the overlap per tick so spawning a golem onto skeletons nudges
    # them aside over a few ticks instead of launching them (CR feel)
    mag = overlap * m[None, :] // ((m[:, None] + m[None, :]) * 4)
    d_safe = jnp.maximum(dist, 1)
    ux = jnp.sign(dx) * (jnp.abs(dx) * mag // d_safe)
    uy = jnp.sign(dy) * (jnp.abs(dy) * mag // d_safe)
    # perfectly coincident bodies: split along x by id order (deterministic)
    coincident = pair & (dist == 0)
    ux = jnp.where(coincident, jnp.where(idx[:, None] < idx[None, :], -mag, mag), ux)
    disp_x = jnp.sum(jnp.where(pair, ux, 0), axis=1)
    disp_y = jnp.sum(jnp.where(pair, uy, 0), axis=1)
    # immovable bodies ignore pushes; cap per-tick displacement to stay stable
    mobile = alive & (C.speed[utype] > 0)
    cap = FP // 4
    disp_x = jnp.clip(jnp.where(mobile, disp_x, 0), -cap, cap)
    disp_y = jnp.clip(jnp.where(mobile, disp_y, 0), -cap, cap)
    nx = jnp.clip(state.u_x + disp_x, 0, W_FP - 1)
    ny = jnp.clip(state.u_y + disp_y, 0, H_FP - 1)

    # tower footprints (ground units only): push fully out of the body circle
    t_body = jnp.where(jnp.asarray(TOWER_IS_CORE) == 1, int(1.8 * FP), int(1.3 * FP))
    tdx = nx[:, None] - jnp.asarray(TOWER_X)[None, :]
    tdy = ny[:, None] - jnp.asarray(TOWER_Y)[None, :]
    tdist = _isqrt(_dist2_16(nx[:, None], ny[:, None],
                             jnp.asarray(TOWER_X)[None, :],
                             jnp.asarray(TOWER_Y)[None, :])) * 16
    t_over = jnp.maximum((t_body + r[:, None]) - tdist, 0)
    t_pair = (alive[:, None] & ~is_air[:, None] & mobile[:, None]
              & (state.tower_hp > 0)[None, :] & (t_over > 0))
    td_safe = jnp.maximum(tdist, 1)
    tux = jnp.sign(tdx) * (jnp.abs(tdx) * t_over // td_safe)
    tuy = jnp.sign(tdy) * (jnp.abs(tdy) * t_over // td_safe)
    # tangential slide (v12): the radial push alone deadlocks a unit whose march
    # points at the tower center (user repro: unit placed behind own princess
    # tower froze at spawn forever). Add a perpendicular component so units walk
    # AROUND footprints; side = sign(tdx) (dead-center breaks toward +x, which
    # is mirror-symmetric under the board's y-flip).
    side = jnp.where(tdx != 0, jnp.sign(tdx), 1)
    sux = side * jnp.sign(-tdy) * (jnp.abs(tdy) * t_over // td_safe)
    suy = side * jnp.sign(tdx) * (jnp.abs(tdx) * t_over // td_safe)
    tux = tux + sux
    tuy = tuy + suy
    nx = jnp.clip(nx + jnp.sum(jnp.where(t_pair, tux, 0), axis=1), 0, W_FP - 1)
    ny = jnp.clip(ny + jnp.sum(jnp.where(t_pair, tuy, 0), axis=1), 0, H_FP - 1)
    nx, ny = _river_guard(state, nx, ny)   # collisions cannot shove units into water
    return state._replace(u_x=nx, u_y=ny)


def _unit_attacks(state: State):
    """Compute simultaneous unit damage. Returns (state, dmg_to_units, dmg_to_towers)."""
    alive = state.u_hp > 0
    utype = state.u_type
    tx, ty = _tgt_pos(state)
    has_tgt = state.u_tgt >= 0
    d2_tgt = _dist2_16(state.u_x, state.u_y, tx, ty)
    ready = (alive & has_tgt & (_delay(state.u_status) == 0) & (state.u_cd == 0)
             & (_stun(state.u_status) == 0)
             & (d2_tgt <= _r2_16(C.range_fp[utype] + _tgt_pad(state))))

    dmg = C.dmg[utype] * jnp.where(_rage(state.u_status) > 0, 13, 10) // 10
    # charged first hit deals double damage, then the run-up resets
    charged_hit = ready & (C.charge[utype] == 1) & (state.u_charge >= CHARGE_FP)
    dmg = jnp.where(charged_hit, dmg * 2, dmg)
    splash2 = _r2_16(C.splash_fp[utype])

    # victims: enemies within splash radius of the TARGET point (radius 0 = target itself)
    d2_vu = _dist2_16(tx[:, None], ty[:, None], state.u_x[None, :], state.u_y[None, :])
    air_ok = (C.targets_air[utype][:, None] == 1) | (C.air[utype][None, :] == 0)
    is_tower_tgt = state.u_tgt >= MAX_UNITS
    direct_u = (~is_tower_tgt[:, None]) & (jnp.arange(MAX_UNITS)[None, :] == state.u_tgt[:, None])
    hit_u = (ready[:, None] & alive[None, :]
             & (state.u_owner[:, None] != state.u_owner[None, :]) & air_ok
             & (direct_u | ((splash2 > 0)[:, None] & (d2_vu <= splash2[:, None]))))
    dmg_units = jnp.sum(hit_u * dmg[:, None], axis=0)

    d2_vt = _dist2_16(tx[:, None], ty[:, None],
                      jnp.asarray(TOWER_X)[None, :], jnp.asarray(TOWER_Y)[None, :])
    direct_t = is_tower_tgt[:, None] \
        & ((state.u_tgt - MAX_UNITS)[:, None] == jnp.arange(N_TOWERS)[None, :])
    hit_t = (ready[:, None] & (state.tower_hp > 0)[None, :]
             & (state.u_owner[:, None] != jnp.asarray(TOWER_OWNER)[None, :])
             & (direct_t | ((splash2 > 0)[:, None] & (d2_vt <= splash2[:, None]))))
    dmg_towers = jnp.sum(hit_t * dmg[:, None], axis=0)

    # on-hit status effects (zappies-stun / ice-wizard-slow families): every victim
    # of an attacker with hit_effect gets the status; stun also resets target locks
    he = C.hit_effect[utype]
    hd = C.hit_dur[utype]
    fired_col = ready[:, None] & hit_u
    stun_in = jnp.max(jnp.where(fired_col & (he[:, None] == EFFECT_STUN),
                                hd[:, None], 0), axis=0)
    slow_in = jnp.max(jnp.where(fired_col & (he[:, None] == EFFECT_SLOW),
                                hd[:, None], 0), axis=0)
    slow, rage = _slow(state.u_status), _rage(state.u_status)
    delay, stun = _delay(state.u_status), _stun(state.u_status)
    stun = jnp.maximum(stun, stun_in)
    slow = jnp.maximum(slow, slow_in)
    tgt = jnp.where(stun_in > 0, -1, state.u_tgt)

    # on-hit stun also freezes TOWERS hit directly or by splash (ice spirit on tower)
    t_stun_in = jnp.max(jnp.where(hit_t & (he[:, None] == EFFECT_STUN),
                                  hd[:, None], 0), axis=0)
    tower_stun = jnp.maximum(state.tower_stun, t_stun_in)

    # v15 Harpooner (fisherman family): each landed hit DRAGS the victim to
    # melee distance of the attacker. Buildings are anchored; river rule holds
    # (a cross-river hook lands the victim on the bank, not in the water).
    hooked_by = fired_col & (C.hook[utype] == 1)[:, None] & direct_u         & (C.archetype[state.u_type] != BUILDING_ARCH)[None, :]
    hooker = jnp.argmax(hooked_by, axis=0)            # attacker per victim
    is_hooked = jnp.any(hooked_by, axis=0)
    hx, hy = state.u_x[hooker], state.u_y[hooker]
    hdx, hdy = state.u_x - hx, state.u_y - hy
    hdist = _isqrt(_dist2_16(state.u_x, state.u_y, hx, hy)) * 16
    keep_fp = jnp.int32(FP)                            # land 1 tile from the hooker
    hmag = jnp.maximum(hdist - keep_fp, 0)
    hux = -jnp.sign(hdx) * (jnp.abs(hdx) * hmag // jnp.maximum(hdist, 1))
    huy = -jnp.sign(hdy) * (jnp.abs(hdy) * hmag // jnp.maximum(hdist, 1))
    nhx = jnp.clip(jnp.where(is_hooked, state.u_x + hux, state.u_x), 0, W_FP - 1)
    nhy = jnp.clip(jnp.where(is_hooked, state.u_y + huy, state.u_y), 0, H_FP - 1)
    # CR: the hook pulls ACROSS the river — a destination inside the water band
    # lands on the hooker's bank instead (never in the water itself)
    in_band = (nhy >= RIVER_LO_FP) & (nhy < RIVER_HI_FP)
    south = hy < RIVER_LO_FP
    bank = jnp.where(south, RIVER_LO_FP - FP // 2, RIVER_HI_FP + FP // 2)
    nhy = jnp.where(is_hooked & in_band, bank, nhy)
    state = state._replace(u_x=nhx, u_y=nhy)

    # kamikaze units (spirits / wall-breakers): die right after their attack lands
    hp = jnp.where(ready & (C.suicide[utype] == 1), 0, state.u_hp)

    cd = jnp.where(ready, C.period[utype], jnp.maximum(state.u_cd - 1, 0))
    charge = jnp.where(ready, 0, state.u_charge)          # attacking ends the charge
    return state._replace(u_cd=cd, u_hp=hp, u_tgt=tgt, tower_stun=tower_stun,
                          u_charge=charge,
                          u_status=_pack(slow, rage, delay, stun)), dmg_units, dmg_towers


def _king_active(state: State):
    """CR rule: the king (core) only fights once damaged or once an own turret falls."""
    hp = state.tower_hp
    p0 = (hp[2] < TOWER_MAX_HP[2]) | (hp[0] == 0) | (hp[1] == 0)
    p1 = (hp[5] < TOWER_MAX_HP[5]) | (hp[3] == 0) | (hp[4] == 0)
    is_core = jnp.asarray(TOWER_IS_CORE) == 1
    owner = jnp.asarray(TOWER_OWNER)
    return ~is_core | jnp.where(owner == 0, p0, p1)


def _tower_attacks(state: State):
    """Towers shoot enemy units (air included) with CR-STICKY targeting: keep the
    locked victim while it lives and stays in range, else retarget nearest.
    CR rule: a tower only engages units that have CROSSED onto its own side of the
    river — never units on the bridge/river rows, never units across the water."""
    alive_t = (state.tower_hp > 0) & _king_active(state)
    d2 = _dist2_16(jnp.asarray(TOWER_X)[:, None], jnp.asarray(TOWER_Y)[:, None],
                   state.u_x[None, :], state.u_y[None, :])
    u_side = _river_side(state.u_y)
    elig = alive_t[:, None] & (state.u_hp > 0)[None, :] \
        & (u_side[None, :] == jnp.asarray(TOWER_OWNER)[:, None]) \
        & (jnp.asarray(TOWER_OWNER)[:, None] != state.u_owner[None, :])
    in_r = elig & (d2 <= _r2_16(jnp.asarray(TOWER_RANGE_ARR))[:, None])

    cur = state.tower_tgt
    ci = jnp.clip(cur, 0, MAX_UNITS - 1)
    keep = (cur >= 0) & in_r[jnp.arange(N_TOWERS), ci]

    d2m = jnp.where(in_r, d2, 2**30)
    nearest = jnp.argmin(d2m, axis=1).astype(jnp.int32)
    has_new = d2m[jnp.arange(N_TOWERS), nearest] < 2**30
    victim = jnp.where(keep, ci, jnp.where(has_new, nearest, -1))

    # CR: towers take a short aim delay on a NEW target before the first shot
    new_tgt = (victim >= 0) & (victim != cur)
    aimed_cd = jnp.where(new_tgt, jnp.maximum(state.tower_cd, TOWER_AIM_TICKS),
                         state.tower_cd)
    fire = (victim >= 0) & (aimed_cd == 0) & alive_t \
        & (state.tower_stun == 0)
    vi = jnp.clip(victim, 0, MAX_UNITS - 1)
    dmg = jnp.zeros(MAX_UNITS, jnp.int32).at[vi].add(
        jnp.where(fire, jnp.asarray(TOWER_DMG), 0))
    cd = jnp.where(fire, jnp.asarray(TOWER_PERIOD_ARR),
                   jnp.maximum(aimed_cd - 1, 0))
    stun = jnp.maximum(state.tower_stun - 1, 0)
    return state._replace(tower_cd=cd, tower_tgt=victim, tower_stun=stun), dmg


def _building_decay(state: State) -> State:
    decay = C.decay[state.u_type]
    hp = jnp.where(state.u_hp > 0, jnp.maximum(state.u_hp - decay, 0), state.u_hp)
    return state._replace(u_hp=hp)


# ---------------------------------------------------------------- step
def to_engine_frame(player, action):
    """Convert a player-centric action [slot, x, y] to the engine frame."""
    slot, x, y = action[0], action[1], action[2]
    ex = jnp.where(player == 1, W - 1 - x, x)
    ey = jnp.where(player == 1, H - 1 - y, y)
    return jnp.stack([slot, ex, ey])


def step(state: State, actions: jax.Array, key: jax.Array | None = None) -> State:
    """actions: (2, 3) int32, PLAYER-CENTRIC [slot 0..3 | 4=noop, x_tile, y_tile] per player.

    `key` is unused in v1 (no stochastic branches); accepted for interface stability.
    """
    del key
    a0 = to_engine_frame(0, actions[0])
    a1 = to_engine_frame(1, actions[1])
    state = _process_action(state, 0, a0)
    state = _process_action(state, 1, a1)

    # pending spells fly; impact when their timer reaches zero
    active = state.spells[:, 0] >= 0
    tl = jnp.where(active, state.spells[:, 4] - 1, state.spells[:, 4])
    state = state._replace(spells=state.spells.at[:, 4].set(tl))
    for k in range(8):
        row = state.spells[k]
        hit = (row[0] >= 0) & (row[4] <= 0)
        state = _apply_spell(state, row[1], row[0], row[2], row[3], hit)
        state = state._replace(
            spells=jnp.where(hit, state.spells.at[k].set(
                jnp.full(5, -1, jnp.int32)), state.spells))

    state = _tick_status(state)
    state = _apply_auras(state)
    state = _acquire_targets(state)
    state = _move(state)
    state = _resolve_collisions(state)

    alive_pre = state.u_hp > 0
    state, dmg_u, dmg_t = _unit_attacks(state)
    state, dmg_u2 = _tower_attacks(state)
    hp = jnp.maximum(state.u_hp - jnp.where(state.u_hp > 0, dmg_u + dmg_u2, 0), 0)
    tower_hp = jnp.maximum(state.tower_hp - dmg_t, 0)

    # death bombs (single pass; bomb-triggered bombs resolve as ordinary deaths):
    # units alive at tick start that are dead now splash death_dmg around their body
    died = alive_pre & (hp <= 0)
    dd = C.death_dmg[state.u_type]
    bomber = died & (dd > 0)
    dr2 = _r2_16(C.death_r_fp[state.u_type])
    d2_v = _dist2_16(state.u_x[:, None], state.u_y[:, None],
                     state.u_x[None, :], state.u_y[None, :])
    boom_u = (bomber[:, None] & (hp > 0)[None, :]
              & (state.u_owner[:, None] != state.u_owner[None, :])
              & (d2_v <= dr2[:, None]))
    hp = jnp.maximum(hp - jnp.sum(boom_u * dd[:, None], axis=0), 0)
    d2_t = _dist2_16(state.u_x[:, None], state.u_y[:, None],
                     jnp.asarray(TOWER_X)[None, :], jnp.asarray(TOWER_Y)[None, :])
    boom_t = (bomber[:, None] & (tower_hp > 0)[None, :]
              & (state.u_owner[:, None] != jnp.asarray(TOWER_OWNER)[None, :])
              & (d2_t <= dr2[:, None]))
    tower_hp = jnp.maximum(tower_hp - jnp.sum(boom_t * dd[:, None], axis=0), 0)
    state = state._replace(u_hp=hp, tower_hp=tower_hp)

    state = _building_decay(state)

    # CR elixir schedule: 1x, 2x in the last 60 s of regulation, 3x in overtime
    regen = jnp.where(state.tick >= TICKS_REG, 3,
                      jnp.where(state.tick >= DOUBLE_TICK, 2, 1))
    energy = jnp.minimum(state.energy + regen, E_MAX)
    return state._replace(energy=energy, tick=state.tick + 1)


# ---------------------------------------------------------------- result
def result(state: State) -> jax.Array:
    """-1 ongoing, 0/1 winner, 2 draw.

    Core destroyed ends the match immediately (both same tick = draw). At regulation
    end and during overtime, any tower-count lead wins (sudden death: the first tower
    to fall in OT breaks the tie). Tied at TICKS_MAX -> CR tiebreaker: the side whose
    LOWEST-hp standing tower is healthier wins; exactly equal = draw."""
    towers = state.tower_hp
    core0_dead = towers[2] == 0
    core1_dead = towers[5] == 0
    dead_of = jnp.array([jnp.sum((towers[3:] == 0)), jnp.sum((towers[:3] == 0))])  # score p0,p1
    lead = jnp.where(dead_of[0] > dead_of[1], RESULT_P0,
                     jnp.where(dead_of[1] > dead_of[0], RESULT_P1, RESULT_DRAW))
    BIG = jnp.int32(10**9)
    min0 = jnp.min(jnp.where(towers[:3] > 0, towers[:3], BIG))
    min1 = jnp.min(jnp.where(towers[3:] > 0, towers[3:], BIG))
    tiebreak = jnp.where(min0 > min1, RESULT_P0,
                         jnp.where(min1 > min0, RESULT_P1, RESULT_DRAW))
    r = jnp.where(
        core0_dead & core1_dead, RESULT_DRAW,
        jnp.where(core1_dead, RESULT_P0,
                  jnp.where(core0_dead, RESULT_P1,
                            jnp.where((state.tick >= TICKS_REG) & (lead != RESULT_DRAW), lead,
                                      jnp.where(state.tick >= TICKS_MAX, tiebreak,
                                                RESULT_ONGOING)))))
    return r.astype(jnp.int32)


# ---------------------------------------------------------------- legal
def legal(state: State, player) -> jax.Array:
    """(5, W, H) bool, PLAYER-CENTRIC. Plane 4 (noop) is always fully legal.

    Capacity uses an 8-slot safety margin (the max bodies a single play can spawn):
    both players' masks are computed from the same pre-tick state, and player 0's
    spawn is processed first — the margin keeps player 1's mask sound regardless of
    what player 0 plays that tick (found by sanity_10k: 72 races in 24M plays)."""
    cards = state.hand[player]                                     # (4,)
    afford = state.energy[player] >= C.cost[cards] * E_UNIT        # (4,)
    spell = C.is_spell[cards] == 1
    anywhere = C.anywhere[cards] == 1
    margin = int(SPAWN_OFF.shape[0])
    cap = jnp.sum(state.u_hp <= 0) >= C.count[cards] + margin      # (4,)
    y = jnp.arange(H)
    x = jnp.arange(W)
    own_half_pf = (y <= 14)                                        # player-frame rows
    # tower pockets (player frame): killing the enemy turret on a lane unlocks
    # that lane's opponent-half rows 17..25
    lane0_open = jnp.where(player == 0, state.tower_hp[3] == 0, state.tower_hp[1] == 0)
    lane1_open = jnp.where(player == 0, state.tower_hp[4] == 0, state.tower_hp[0] == 0)
    pocket_rows = (y >= 17) & (y <= 23)                            # (H,)
    lane_open = jnp.where(x < 9, lane0_open, lane1_open)           # (W,)
    pocket = lane_open[:, None] & pocket_rows[None, :]             # (W, H)
    ground_ok = own_half_pf[None, None, :] | pocket[None, :, :]
    tile_ok = (spell[:, None, None]
               | (anywhere & cap)[:, None, None]
               | (ground_ok & cap[:, None, None]))
    slot_mask = afford[:, None, None] & tile_ok                    # (4, 1|W, H) -> broadcast
    slot_mask = jnp.broadcast_to(slot_mask, (HAND, W, H))
    noop = jnp.ones((1, W, H), bool)
    return jnp.concatenate([slot_mask, noop], axis=0)


# ---------------------------------------------------------------- observe
def observe(state: State, player) -> Obs:
    """Player-centric observation. Hides the opponent's hand/queue (imperfect info)."""
    alive = state.u_hp > 0
    utype = state.u_type
    arch = C.archetype[utype]
    own = state.u_owner == player

    # player-frame tile coords (180-degree rotation for player 1)
    px = jnp.where(player == 1, W_FP - 1 - state.u_x, state.u_x) // FP
    py = jnp.where(player == 1, H_FP - 1 - state.u_y, state.u_y) // FP
    px = jnp.clip(px, 0, W - 1)
    py = jnp.clip(py, 0, H - 1)

    plane = jnp.where(own, arch, arch + 12)
    spatial = jnp.zeros((H, W, OBS_C), jnp.float32)
    spatial = spatial.at[py, px, plane].add(jnp.where(alive, 1.0, 0.0))
    hp_plane = jnp.where(own, 24, 25)
    spatial = spatial.at[py, px, hp_plane].add(
        jnp.where(alive, state.u_hp.astype(jnp.float32) / 2000.0, 0.0))

    t_own = jnp.asarray(TOWER_OWNER) == player
    tx = jnp.where(player == 1, W_FP - 1 - jnp.asarray(TOWER_X), jnp.asarray(TOWER_X)) // FP
    ty = jnp.where(player == 1, H_FP - 1 - jnp.asarray(TOWER_Y), jnp.asarray(TOWER_Y)) // FP
    t_plane = jnp.where(t_own, 26, 27)
    t_frac = state.tower_hp.astype(jnp.float32) / jnp.asarray(TOWER_MAX_HP, jnp.float32)
    spatial = spatial.at[ty, tx, t_plane].add(t_frac)

    opp = 1 - player
    hand_oh = jax.nn.one_hot(state.hand[player], N_CARDS).reshape(-1)
    queue_oh = jax.nn.one_hot(state.queue[player], N_CARDS).reshape(-1)
    own_t = jnp.where(player == 1, t_frac[3:], t_frac[:3])
    opp_t = jnp.where(player == 1, t_frac[:3], t_frac[3:])
    vector = jnp.concatenate([
        jnp.array([state.energy[player] / E_MAX, state.energy[opp] / E_MAX,
                   state.tick / TICKS_MAX, (state.tick >= TICKS_REG) * 1.0], jnp.float32),
        own_t, opp_t, hand_oh, queue_oh,
    ])
    return Obs(spatial=spatial, vector=vector)
