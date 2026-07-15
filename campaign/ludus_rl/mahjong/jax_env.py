"""True vmappable JAX environment for Chinese Standard Mahjong (MCR) — v1.

This is a **v2 training-accelerator track**, NOT a replacement for the
oracle-validated pure-Python engine (``mahjong/engine.py`` +
``mahjong/validate_adapter.py:MyEngine``, 12,288/12,288).  Nothing here touches
those; this is a new, separate, GPU-parallel RL-rollout env that reproduces the
SAME transitions as ``MyEngine`` so thousands of games can be ``jax.vmap``-ed.

It disproves "sequential and branchy games don't vmap" the same way pgx does
chess/Go/Shogi: the whole game state is a pytree of fixed-size tensors and the
step function is a masked ``jax.lax.switch`` over the phase — no Python control
flow inside the jitted step.

============================================================================
STATE LAYOUT  (``State`` NamedTuple — all fixed-size, vmappable)
============================================================================
  wall              int8 [136]   tile codes 0..33; seat s' segment = wall[34s:34s+34]
  draw_ptr          int32[4]     tiles left drawable in each seat's segment (start 21).
                                 Next draw = wall[34s + draw_ptr[s]-1]; back-to-front.
  hands             int8 [4,34]  concealed-hand counts per seat
  plays             int8 [4,34]  every tile each seat has DISCARDED (never decremented;
                                 mirrors champion feature.py `history`, for obs)
  discards          int8 [4,34]  LIVE discard counts (decremented when a discard is
                                 claimed) — mirrors validate_adapter self.discards
  melds             int8 [4,4,3] up to 4 melds/seat: (type, tile, offer)
                                 type 0=none 1=CHI 2=PENG 3=GANG(melded/concealed/added)
                                 offer: CHI position 1..3; PENG/GANG rel-seat (0=concealed)
  meld_cnt          int32[4]     melds declared per seat
  last_discard_tile int32        live unclaimed discard tile idx (-1 none)
  last_discard_seat int32        seat that made the live discard (-1)
  last_draw_tile    int32        tile just drawn (for zimo win-tile / about-kong)
  phase             int32        phase enum (see below)
  seat_to_act       int32        seat whose draw/kong-replacement is pending
  kong              int32        about-kong flag for the next zimo/draw
  quan              int32        prevalent (round) wind 0..3
  done              int32        1 when terminal
  winner..ending    int32        terminal context (filled at HU/HUANG)
  fan               int32        terminal fan total (filled by the callback scorer)
  scores            int32[4]     terminal per-seat MCR score
  last_ev_a/p/t/chi int32        LAST judge event (action enum / player / tile / chi-mid)
                                 -> the replay validator reads these to match the oracle
                                 display stream (action/player/tile/tileCHI).

============================================================================
PHASE MACHINE  (step = lax.switch on `phase`; each branch a masked update)
============================================================================
  P_DRAW      (0)  a DRAW event was emitted; `seat_to_act` (the drawer) must act:
                   {Hu(zimo) | Play t | AnGang t | BuGang t}.
  P_DISCARD   (1)  a PLAY/PENG/CHI event left a live discard; the 3 non-discarders
                   each pick a claim {Pass|Hu|Peng|Gang|Chi}; resolved by ONE
                   priority-weighted argmax (NOT recursive). A granted Peng/Chi
                   emits that seat's PENG/CHI event (its follow-up discard is
                   carried in the action) and stays in P_DISCARD for exactly one
                   more resolution round. A granted melded Gang -> P_GANG_DRAW.
  P_GANG_DRAW (2)  replacement draw for `seat_to_act` after a (An)Gang/melded Gang;
                   deterministic -> emits DRAW -> P_DRAW.
  P_BUGANG    (3)  a BUGANG event was emitted; the 3 non-drawers may rob-the-kong
                   (qianggang) with Hu. None -> replacement DRAW -> P_DRAW.
  P_TERMINAL  (4)  done; identity branch (finished vmap lanes no-op).

Claim priority (`_PRIO`): Hu(3) > Peng/Gang(2) > Chi(1); ties broken by nearest
downstream seat ((ss-discarder)%4). Implemented as argmax of `prio*8 - dist`.

============================================================================
FAN SCORING — v1 HYBRID
============================================================================
The whole game flow is pure-tensor & jittable. The fan/score is computed by
the existing ground-truth ``MahjongGB.MahjongFanCalculator`` — called via
``jax.pure_callback`` ONLY at Hu events (`score_state`), which are a few % of
steps. ~95% of the throughput win, zero correctness risk (the callback is the
same PyMahjongGB the oracle uses). ``replay_game`` calls the same host function
directly at the terminal.

v2 FUTURE WORK (not built here): a fully tensorized fan scorer (81 MCR fans as
lookup tables / count-pattern matchers) to drop the callback and score entirely
on-device — the last ~5% of throughput and the piece needed to score dense
self-play Hu without a host round-trip.

============================================================================
ACTION SPACE  (same 235-space as mahjong/rl_env.py, champion feature.py)
============================================================================
  0 Pass | 1 Hu | 2..35 Play t | 36..98 Chi | 99..132 Peng | 133..166 Gang
  167..200 AnGang | 201..234 BuGang
step() consumes `actions` int32[4,2]: column0 = 235-code per seat (the primary
decision at the current phase; non-acting seats ignored), column1 = follow-up
discard tile idx (0..33) used ONLY when that seat is the granted Peng/Chi
claimer (the oracle bundles the claim + its follow discard into one event).
"""

from functools import partial
import typing

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

try:
    from MahjongGB import MahjongFanCalculator
except Exception:  # pragma: no cover - only on the test box
    MahjongFanCalculator = None

# ---------------------------------------------------------------- tile tables
# champion order: W1-9(0-8) T1-9(9-17) B1-9(18-26) F1-4(27-30) J1-3(31-33)
TILE_LIST = ([f"W{n}" for n in range(1, 10)] + [f"T{n}" for n in range(1, 10)]
             + [f"B{n}" for n in range(1, 10)] + [f"F{n}" for n in range(1, 5)]
             + [f"J{n}" for n in range(1, 4)])
TILE2IDX = {c: i for i, c in enumerate(TILE_LIST)}
N_ACT = 235

# action-space offsets (champion feature.py)
A_PASS, A_HU, A_PLAY, A_CHI, A_PENG, A_GANG, A_ANGANG, A_BUGANG = \
    0, 1, 2, 36, 99, 133, 167, 201

# phase enum
P_DRAW, P_DISCARD, P_GANG_DRAW, P_BUGANG, P_TERMINAL = 0, 1, 2, 3, 4
# event enum (last_ev_a)
E_INIT, E_DEAL, E_DRAW, E_PLAY, E_PENG, E_CHI, E_GANG, E_BUGANG, E_HU, E_HUANG = \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9
EV_NAME = {E_DRAW: "DRAW", E_PLAY: "PLAY", E_PENG: "PENG", E_CHI: "CHI",
           E_GANG: "GANG", E_BUGANG: "BUGANG", E_HU: "HU", E_HUANG: "HUANG"}
# claim-type prio: [pass, hu, peng, gang, chi]
_PRIO = jnp.array([0, 3, 2, 2, 1], dtype=jnp.int32)


class State(typing.NamedTuple):
    wall: jnp.ndarray
    draw_ptr: jnp.ndarray
    hands: jnp.ndarray
    plays: jnp.ndarray
    discards: jnp.ndarray
    melds: jnp.ndarray
    meld_cnt: jnp.ndarray
    last_discard_tile: jnp.ndarray
    last_discard_seat: jnp.ndarray
    last_draw_tile: jnp.ndarray
    phase: jnp.ndarray
    seat_to_act: jnp.ndarray
    kong: jnp.ndarray
    quan: jnp.ndarray
    done: jnp.ndarray
    winner: jnp.ndarray
    win_tile: jnp.ndarray
    discarder: jnp.ndarray
    is_self: jnp.ndarray
    about_kong: jnp.ndarray
    wall_last: jnp.ndarray
    ending: jnp.ndarray          # 0 none 1 zimo 2 ron 3 huang
    fan: jnp.ndarray
    scores: jnp.ndarray
    last_ev_a: jnp.ndarray
    last_ev_p: jnp.ndarray
    last_ev_t: jnp.ndarray
    last_ev_chi: jnp.ndarray


def _i32(x):
    return jnp.asarray(x, dtype=jnp.int32)


def _tree_select(pred, a, b):
    """Select whole pytree `a` if pred else `b` (pred scalar)."""
    return jax.tree_util.tree_map(lambda x, y: jnp.where(pred, x, y), a, b)


# ------------------------------------------------------------------- draw op
def _do_draw(state, seat, kong):
    """Perform seat's draw (or HUANG if its segment is empty). Emits DRAW/HUANG,
    lands in P_DRAW (or terminal)."""
    seat = _i32(seat)
    empty = state.draw_ptr[seat] == 0
    pos = 34 * seat + state.draw_ptr[seat] - 1
    tile = state.wall[jnp.clip(pos, 0, 135)].astype(jnp.int32)
    drawn = state._replace(
        draw_ptr=state.draw_ptr.at[seat].add(-1),
        hands=state.hands.at[seat, tile].add(1),
        last_draw_tile=tile,
        phase=_i32(P_DRAW), seat_to_act=seat, kong=_i32(kong),
        last_ev_a=_i32(E_DRAW), last_ev_p=seat, last_ev_t=tile,
        last_ev_chi=_i32(-1))
    huang = state._replace(
        phase=_i32(P_TERMINAL), done=_i32(1), ending=_i32(3),
        last_ev_a=_i32(E_HUANG), last_ev_p=_i32(-1), last_ev_t=_i32(-1),
        last_ev_chi=_i32(-1))
    return _tree_select(empty, huang, drawn)


def _set_hu(state, winner, win_tile, is_self, about_kong, discarder, wall_last,
            ending):
    return state._replace(
        phase=_i32(P_TERMINAL), done=_i32(1), winner=_i32(winner),
        win_tile=_i32(win_tile), is_self=_i32(is_self),
        about_kong=_i32(about_kong), discarder=_i32(discarder),
        wall_last=_i32(wall_last), ending=_i32(ending),
        last_ev_a=_i32(E_HU), last_ev_p=_i32(winner), last_ev_t=_i32(-1),
        last_ev_chi=_i32(-1))


def _add_meld(state, seat, mtype, tile, offer):
    mc = state.meld_cnt[seat]
    melds = state.melds.at[seat, mc, 0].set(jnp.int8(mtype))
    melds = melds.at[seat, mc, 1].set(jnp.int8(tile))
    melds = melds.at[seat, mc, 2].set(jnp.int8(offer))
    return state._replace(melds=melds, meld_cnt=state.meld_cnt.at[seat].add(1))


# ================================================================ DRAW branch
def _dr_hu(state, seat, a, aux):
    wl = (state.draw_ptr[(seat + 1) % 4] == 0).astype(jnp.int32)
    return _set_hu(state, seat, state.last_draw_tile, 1, state.kong, -1, wl, 1)


def _dr_play(state, seat, a, aux):
    tile = a - A_PLAY
    return state._replace(
        hands=state.hands.at[seat, tile].add(-1),
        discards=state.discards.at[seat, tile].add(1),
        plays=state.plays.at[seat, tile].add(1),
        last_discard_tile=tile, last_discard_seat=seat,
        phase=_i32(P_DISCARD),
        last_ev_a=_i32(E_PLAY), last_ev_p=seat, last_ev_t=tile,
        last_ev_chi=_i32(-1))


def _dr_angang(state, seat, a, aux):
    tile = a - A_ANGANG
    st = state._replace(hands=state.hands.at[seat, tile].add(-4))
    st = _add_meld(st, seat, 3, tile, 0)          # concealed kong (offer 0)
    return st._replace(
        phase=_i32(P_GANG_DRAW), seat_to_act=seat, kong=_i32(1),
        last_ev_a=_i32(E_GANG), last_ev_p=seat, last_ev_t=tile,
        last_ev_chi=_i32(-1))


def _dr_bugang(state, seat, a, aux):
    tile = a - A_BUGANG
    m = state.melds[seat]
    up = (m[:, 0] == 2) & (m[:, 1] == tile)       # matching PENG meld
    new_type = jnp.where(up, jnp.int8(3), m[:, 0])
    melds = state.melds.at[seat, :, 0].set(new_type)
    return state._replace(
        hands=state.hands.at[seat, tile].add(-1), melds=melds,
        phase=_i32(P_BUGANG), seat_to_act=seat, kong=_i32(1),
        last_ev_a=_i32(E_BUGANG), last_ev_p=seat, last_ev_t=tile,
        last_ev_chi=_i32(-1))


def _draw_branch(state, actions):
    seat = state.seat_to_act
    a = actions[seat, 0]
    aux = actions[seat, 1]
    cat = jnp.where(a == A_HU, 0,
                    jnp.where(a < A_CHI, 1, jnp.where(a < A_BUGANG, 2, 3)))
    return lax.switch(cat, [_dr_hu, _dr_play, _dr_angang, _dr_bugang],
                      state, seat, a, aux)


# ============================================================= DISCARD branch
def _decode_claim(a):
    return jnp.where(a == A_HU, 1,
             jnp.where((a >= A_PENG) & (a < A_GANG), 2,
               jnp.where((a >= A_GANG) & (a < A_ANGANG), 3,
                 jnp.where((a >= A_CHI) & (a < A_PENG), 4, 0))))


def _cl_none(state, disc, live, winner, actions):
    nxt = (disc + 1) % 4
    return _do_draw(state._replace(kong=_i32(0)), nxt, 0)


def _cl_hu(state, disc, live, winner, actions):        # ron
    wl = (state.draw_ptr[(disc + 1) % 4] == 0).astype(jnp.int32)
    st = state._replace(discards=state.discards.at[disc, live].add(-1))
    return _set_hu(st, winner, live, 0, 0, disc, wl, 2)


def _cl_peng(state, disc, live, winner, actions):
    ws = winner
    follow = actions[ws, 1]
    st = state._replace(
        hands=state.hands.at[ws, live].add(-2),
        discards=state.discards.at[disc, live].add(-1))
    st = _add_meld(st, ws, 2, live, (disc - ws) % 4)
    st = st._replace(
        hands=st.hands.at[ws, follow].add(-1),
        discards=st.discards.at[ws, follow].add(1),
        plays=st.plays.at[ws, follow].add(1),
        last_discard_tile=follow, last_discard_seat=ws,
        phase=_i32(P_DISCARD), kong=_i32(0),
        last_ev_a=_i32(E_PENG), last_ev_p=ws, last_ev_t=follow,
        last_ev_chi=_i32(-1))
    return st


def _cl_gang(state, disc, live, winner, actions):      # melded kong of a discard
    ws = winner
    st = state._replace(
        hands=state.hands.at[ws, live].add(-3),
        discards=state.discards.at[disc, live].add(-1))
    st = _add_meld(st, ws, 3, live, (disc - ws) % 4)
    return st._replace(
        phase=_i32(P_GANG_DRAW), seat_to_act=ws, kong=_i32(1),
        last_ev_a=_i32(E_GANG), last_ev_p=ws, last_ev_t=live,
        last_ev_chi=_i32(-1))


def _cl_chi(state, disc, live, winner, actions):
    ws = (disc + 1) % 4
    a = actions[ws, 0]
    follow = actions[ws, 1]
    t = (a - A_CHI) // 3
    suit = t // 7
    midnum = t % 7 + 2                # 2..8
    mid = suit * 9 + (midnum - 1)     # tile idx of the run middle
    r0, r1, r2 = mid - 1, mid, mid + 1
    is0 = r0 == live
    is2 = r2 == live
    f1 = jnp.where(is0, r1, r0)
    f2 = jnp.where(is2, r1, r2)
    live_num = live % 9 + 1
    offer = live_num - midnum + 2     # 1..3 position of claimed tile
    st = state._replace(
        hands=state.hands.at[ws, f1].add(-1),
        discards=state.discards.at[disc, live].add(-1))
    st = st._replace(hands=st.hands.at[ws, f2].add(-1))
    st = _add_meld(st, ws, 1, mid, offer)
    return st._replace(
        hands=st.hands.at[ws, follow].add(-1),
        discards=st.discards.at[ws, follow].add(1),
        plays=st.plays.at[ws, follow].add(1),
        last_discard_tile=follow, last_discard_seat=ws,
        phase=_i32(P_DISCARD), kong=_i32(0),
        last_ev_a=_i32(E_CHI), last_ev_p=ws, last_ev_t=follow,
        last_ev_chi=mid)


def _discard_branch(state, actions):
    disc = state.last_discard_seat
    live = state.last_discard_tile
    codes = actions[:, 0]
    ctype = jax.vmap(_decode_claim)(codes)                 # [4]
    seats = jnp.arange(4)
    dist = (seats - disc) % 4
    prio = _PRIO[ctype]
    score = prio * 8 - dist
    score = jnp.where(seats == disc, -1000, score)
    winner = jnp.argmax(score)
    wprio = prio[winner]
    wtype = ctype[winner]
    eff = jnp.where(wprio == 0, 0, wtype)                  # 0 none/rotate
    return lax.switch(eff, [_cl_none, _cl_hu, _cl_peng, _cl_gang, _cl_chi],
                      state, disc, live, winner, actions)


# ========================================================== GANG_DRAW branch
def _gangdraw_branch(state, actions):
    return _do_draw(state, state.seat_to_act, 1)


# ============================================================= BUGANG branch
def _bugang_branch(state, actions):
    turn = state.seat_to_act
    bt = state.last_ev_t
    codes = actions[:, 0]
    seats = jnp.arange(4)
    is_hu = (codes == A_HU) & (seats != turn)
    score = jnp.where(is_hu, 8 - ((seats - turn) % 4), -1000)
    winner = jnp.argmax(score)
    robbed = jnp.any(is_hu)
    wl = (state.draw_ptr[(turn + 1) % 4] == 0).astype(jnp.int32)
    hu_state = _set_hu(state, winner, bt, 0, 1, turn, wl, 2)
    draw_state = _do_draw(state, turn, 1)
    return _tree_select(robbed, hu_state, draw_state)


# ---------------------------------------------------------------------- step
@jax.jit
def step(state, actions):
    """Pure, jittable, vmappable transition. `actions` int32[4,2]."""
    actions = jnp.asarray(actions, dtype=jnp.int32)
    return lax.switch(state.phase,
                      [_draw_branch, _discard_branch, _gangdraw_branch,
                       _bugang_branch, lambda s, a: s],
                      state, actions)


# --------------------------------------------------------------------- reset
def reset(wall, quan):
    """wall: int8[136] tile codes 0..33 (champion order); quan int. Deals 13/seat
    (last-13-reversed of each 34-tile segment) then performs seat 0's first draw,
    returning the State at that first DRAW event (phase P_DRAW)."""
    wall = jnp.asarray(wall, dtype=jnp.int8)
    hands = jnp.zeros((4, 34), dtype=jnp.int8)
    for s in range(4):
        for i in range(13):
            hands = hands.at[s, wall[34 * s + 33 - i]].add(1)
    base = State(
        wall=wall, draw_ptr=jnp.full((4,), 21, jnp.int32), hands=hands,
        plays=jnp.zeros((4, 34), jnp.int8), discards=jnp.zeros((4, 34), jnp.int8),
        melds=jnp.zeros((4, 4, 3), jnp.int8), meld_cnt=jnp.zeros((4,), jnp.int32),
        last_discard_tile=_i32(-1), last_discard_seat=_i32(-1),
        last_draw_tile=_i32(-1), phase=_i32(P_DRAW), seat_to_act=_i32(0),
        kong=_i32(0), quan=_i32(quan), done=_i32(0), winner=_i32(-1),
        win_tile=_i32(-1), discarder=_i32(-1), is_self=_i32(0),
        about_kong=_i32(0), wall_last=_i32(0), ending=_i32(0), fan=_i32(0),
        scores=jnp.zeros((4,), jnp.int32),
        last_ev_a=_i32(E_DEAL), last_ev_p=_i32(-1), last_ev_t=_i32(-1),
        last_ev_chi=_i32(-1))
    return _do_draw(base, 0, 0)


# ----------------------------------------------------------------- legal_mask
def _draw_mask_row(state, seat):
    row = jnp.zeros((N_ACT,), dtype=bool)
    hand = state.hands[seat]
    tiles = jnp.arange(34)
    # Play any held tile
    row = row.at[A_PLAY + tiles].set(hand > 0)
    drawable = state.draw_ptr[seat] > 0
    wall_last = state.draw_ptr[(seat + 1) % 4] == 0
    ok_kong = drawable & (~wall_last)
    has_slot = state.meld_cnt[seat] < 4        # a 5th meld is impossible (issue #3)
    # AnGang: 4 in hand (consumes a new meld slot -> needs a free slot)
    row = row.at[A_ANGANG + tiles].set((hand == 4) & ok_kong & has_slot)
    # BuGang: an existing PENG whose tile is in hand
    m = state.melds[seat]
    peng_tile = jnp.where(m[:, 0] == 2, m[:, 1], -1)
    bug = jnp.zeros((34,), dtype=bool)
    for k in range(4):
        pt = peng_tile[k]
        bug = bug | ((tiles == pt) & (hand[jnp.clip(pt, 0, 33)] > 0))
    row = row.at[A_BUGANG + tiles].set(bug & ok_kong)
    # (Hu is intentionally excluded from the random-legal mask — it needs a
    #  fan-calc callback; the random rollout policy never declares Hu.)
    return row


def _claim_mask_row(state, seat):
    disc = state.last_discard_seat
    live = state.last_discard_tile
    row = jnp.zeros((N_ACT,), dtype=bool)
    row = row.at[A_PASS].set(True)
    hand = state.hands[seat]
    is_other = seat != disc
    wall_last = state.draw_ptr[(disc + 1) % 4] == 0
    has_slot = state.meld_cnt[seat] < 4        # Peng/Gang/Chi each take a new slot
    lc = hand[jnp.clip(live, 0, 33)]
    can_peng = is_other & (lc >= 2) & (~wall_last) & has_slot
    can_gang = is_other & (lc == 3) & (state.draw_ptr[seat] > 0) & (~wall_last) & has_slot
    row = row.at[A_PENG + jnp.clip(live, 0, 33)].set(
        row[A_PENG + jnp.clip(live, 0, 33)] | can_peng)
    row = row.at[A_GANG + jnp.clip(live, 0, 33)].set(
        row[A_GANG + jnp.clip(live, 0, 33)] | can_gang)
    # Chi: only next seat, suited
    is_next = seat == (disc + 1) % 4
    suited = live < 27
    suit = live // 9
    num = live % 9 + 1
    for (need_lo, midnum, off) in ((num, num + 1, 0), (num - 1, num, 1),
                                   (num - 2, num - 1, 2)):
        # claimed at position `off` of run centred on midnum; fillers are the
        # other two of {midnum-1,midnum,midnum+1}
        mid = suit * 9 + (midnum - 1)
        r0, r1, r2 = mid - 1, mid, mid + 1
        valid = is_next & suited & (midnum >= 2) & (midnum <= 8)
        # fillers = run tiles != live
        f1 = jnp.where(r0 == live, r1, r0)
        f2 = jnp.where(r2 == live, r1, r2)
        have = (hand[jnp.clip(f1, 0, 33)] > 0) & (hand[jnp.clip(f2, 0, 33)] > 0)
        # both fillers must be within the same suit range
        in_suit = (f1 >= suit * 9) & (f1 < suit * 9 + 9) & \
                  (f2 >= suit * 9) & (f2 < suit * 9 + 9)
        code = A_CHI + (suit * 7 + (midnum - 2)) * 3 + off
        ok = valid & have & in_suit & (~wall_last) & has_slot
        row = row.at[jnp.clip(code, 0, N_ACT - 1)].set(
            row[jnp.clip(code, 0, N_ACT - 1)] | ok)
    return row


def legal_mask(state):
    """bool[4,235] per-seat legality for the CURRENT phase's decision (Hu
    excluded — see note). Non-acting seats get Pass only. Random rollout policy
    samples from this."""
    pass_only = jnp.zeros((4, N_ACT), dtype=bool).at[:, A_PASS].set(True)
    draw_rows = pass_only.at[state.seat_to_act].set(
        _draw_mask_row(state, state.seat_to_act))
    claim_rows = jax.vmap(lambda s: _claim_mask_row(state, s))(jnp.arange(4))
    return lax.switch(state.phase,
                      [lambda: draw_rows, lambda: claim_rows,
                       lambda: pass_only, lambda: pass_only,
                       lambda: pass_only])


# --------------------------------------------------------------- observation
def obs(state, seat):
    """(38,4,9) float32 observation for `seat`, champion feature.py layout
    (hidden info masked to the seat's own hand). Planes:
      0 seat wind, 1 prevalent wind, 2-5 own hand (count-encoded),
      6-21 discards (4 planes x 4 players, self-relative),
      22-37 melds (4 planes x 4 players, tiles laid flat)."""
    o = jnp.zeros((38, 36), dtype=jnp.float32)
    o = o.at[0, 27 + seat].set(1.0)
    o = o.at[1, 27 + state.quan].set(1.0)
    hand = state.hands[seat]
    for k in range(4):
        o = o.at[2 + k, :34].set((hand > k).astype(jnp.float32))
    for i in range(4):
        p = (seat + i) % 4
        pl = state.plays[p]
        for k in range(4):
            o = o.at[6 + 4 * i + k, :34].set((pl > k).astype(jnp.float32))
    # melds -> flat tile counts per player
    for i in range(4):
        p = (seat + i) % 4
        cnt = jnp.zeros((34,), jnp.float32)
        for mslot in range(4):
            mt = state.melds[p, mslot, 0]
            tl = state.melds[p, mslot, 1]
            # CHI -> tl-1,tl,tl+1 ; PENG -> 3x tl ; GANG -> 4x tl
            def add_tiles(cnt):
                is_chi = mt == 1
                lo = jnp.clip(tl - 1, 0, 33)
                hi = jnp.clip(tl + 1, 0, 33)
                cnt = cnt + jnp.where(is_chi, jax.nn.one_hot(lo, 34), 0.)
                cnt = cnt + jnp.where(is_chi, jax.nn.one_hot(jnp.clip(tl, 0, 33), 34), 0.)
                cnt = cnt + jnp.where(is_chi, jax.nn.one_hot(hi, 34), 0.)
                nrep = jnp.where(mt == 2, 3., jnp.where(mt == 3, 4., 0.))
                cnt = cnt + nrep * jax.nn.one_hot(jnp.clip(tl, 0, 33), 34)
                return cnt
            cnt = jnp.where(mt > 0, add_tiles(cnt), cnt)
        for k in range(4):
            o = o.at[22 + 4 * i + k, :34].set((cnt > k).astype(jnp.float32))
    return o.reshape((38, 4, 9))


# =========================================================== full-game driver
def random_action_fn(state, key):
    """Random-LEGAL policy: gumbel-argmax over legal_mask per seat + a legal
    follow-up discard (a held tile != the live tile). int32[4,2]."""
    mask = legal_mask(state)
    g = jax.random.gumbel(key, (4, N_ACT))
    logits = jnp.where(mask, 0.0, -1e9) + g
    primary = jnp.argmax(logits, axis=1).astype(jnp.int32)
    live = state.last_discard_tile
    hh = jnp.where(jnp.arange(34)[None, :] == live, 0, state.hands)
    aux = jnp.argmax(hh, axis=1).astype(jnp.int32)
    return jnp.stack([primary, aux], axis=1)


def run_game(wall, quan, key, action_fn=random_action_fn, cap=300):
    """Play one full game with lax.while_loop. Returns (final_state, n_steps)."""
    state = reset(wall, quan)

    def cond(carry):
        st, k, n = carry
        return (st.done == 0) & (n < cap)

    def body(carry):
        st, k, n = carry
        k, sub = jax.random.split(k)
        acts = action_fn(st, sub)
        return step(st, acts), k, n + 1

    state, _, n = lax.while_loop(cond, body, (state, key, jnp.int32(0)))
    return state, n


v_run = jax.jit(jax.vmap(run_game, in_axes=(0, None, 0)),
                static_argnums=())


# ============================================ v1 hybrid fan-scorer (callback)
def _host_fan(melds, meld_cnt, hands, win_tile, is_self, about_kong,
              wall_last, is_4th, seat, quan):
    """Host-side PyMahjongGB fan total for a Hu state (numpy inputs)."""
    pack = []
    kmap = {1: "CHI", 2: "PENG", 3: "GANG"}
    for m in range(int(meld_cnt)):
        mt, tl, off = int(melds[m, 0]), int(melds[m, 1]), int(melds[m, 2])
        pack.append((kmap[mt], TILE_LIST[tl], off))
    hand = []
    for idx in range(34):
        hand += [TILE_LIST[idx]] * int(hands[idx])
    wtile = TILE_LIST[int(win_tile)]
    if is_self:
        hand.remove(wtile)
    try:
        res = MahjongFanCalculator(
            pack=tuple(pack), hand=tuple(hand), winTile=wtile, flowerCount=0,
            isSelfDrawn=bool(is_self), is4thTile=bool(is_4th),
            isAboutKong=bool(about_kong), isWallLast=bool(wall_last),
            seatWind=int(seat), prevalentWind=int(quan), verbose=True)
    except Exception:
        return np.int32(-1)          # not a winning hand -> sentinel (issue #4)
    return np.int32(sum(p * c for p, c, *_ in res))


def _is_4th(state, winner, win_tile):
    """Count of win_tile visible in all discards+melds == 3 (numpy state)."""
    vis = int(np.asarray(state.discards)[:, win_tile].sum())
    melds = np.asarray(state.melds)
    mc = np.asarray(state.meld_cnt)
    for s in range(4):
        for m in range(int(mc[s])):
            mt, tl = int(melds[s, m, 0]), int(melds[s, m, 1])
            if mt == 1:                       # CHI
                for t in (tl - 1, tl, tl + 1):
                    vis += (t == win_tile)
            elif mt == 2:
                vis += 3 * (tl == win_tile)
            elif mt == 3:
                vis += 4 * (tl == win_tile)
    return vis == 3


def host_score(state):
    """Compute (fan_total, scores[4]) for a terminal State on the host, using
    the ground-truth PyMahjongGB — the v1 hybrid scorer. Returns (fan, scores).
    For ending=zimo/ron; draw -> (0, [0,0,0,0])."""
    ending = int(state.ending)
    if ending == 3 or ending == 0:
        return 0, [0, 0, 0, 0]
    winner = int(state.winner)
    win_tile = int(state.win_tile)
    is_self = int(state.is_self)
    is4 = _is_4th(state, winner, win_tile)
    fan = int(_host_fan(np.asarray(state.melds)[winner],
                        int(np.asarray(state.meld_cnt)[winner]),
                        np.asarray(state.hands)[winner], win_tile, is_self,
                        int(state.about_kong), int(state.wall_last), is4,
                        winner, int(state.quan)))
    scores = [0, 0, 0, 0]
    if fan < 8:                                # ILLEGAL Hu (issue #4): a hand that
        # does not win or scores < the 8-fan minimum. The oracle never records
        # this (recorded Hu are always legal), so replay is unaffected — but an
        # RL agent could declare it. Forfeit: the declarer is PENALISED, never
        # paid. Zero-sum. This closes the reward-hack at scoring time (no
        # per-decision callback needed; the fan callback fires only when Hu is
        # actually declared).
        for s in range(4):
            scores[s] = -24 if s == winner else 8
        return -1, scores
    if ending == 1:                            # zimo
        for s in range(4):
            scores[s] = 3 * (8 + fan) if s == winner else -(8 + fan)
    else:                                      # ron / qianggang
        disc = int(state.discarder)
        for s in range(4):
            if s == winner:
                scores[s] = 24 + fan
            elif s == disc:
                scores[s] = -(8 + fan)
            else:
                scores[s] = -8
    return fan, scores


@jax.jit
def score_state(state):
    """Jittable v1 hybrid scorer: fills state.fan/scores via jax.pure_callback
    to PyMahjongGB, ONLY meaningful at a Hu terminal. Demonstrates the callback
    living inside a jitted flow (the fused-rollout scoring hook)."""
    def cb(melds, meld_cnt, hands, win_tile, is_self, about_kong, wall_last,
           seat, quan, all_disc, all_melds, all_mc):
        # is4thTile (issue #4): win_tile visible 3x across all discards + melds
        wt = int(win_tile)
        vis = int(np.asarray(all_disc)[:, wt].sum())
        am, amc = np.asarray(all_melds), np.asarray(all_mc)
        for s in range(4):
            for m in range(int(amc[s])):
                mt, tl = int(am[s, m, 0]), int(am[s, m, 1])
                if mt == 1:
                    for t in (tl - 1, tl, tl + 1):
                        vis += (t == wt)
                elif mt == 2:
                    vis += 3 * (tl == wt)
                elif mt == 3:
                    vis += 4 * (tl == wt)
        return _host_fan(melds, meld_cnt, hands, win_tile, is_self, about_kong,
                         wall_last, np.int32(vis == 3), seat, quan)
    winner = state.winner
    is_win = (state.ending == 1) | (state.ending == 2)
    w = jnp.clip(winner, 0, 3)
    fan = jax.pure_callback(
        cb, jax.ShapeDtypeStruct((), jnp.int32),
        state.melds[w], state.meld_cnt[w], state.hands[w], state.win_tile,
        state.is_self, state.about_kong, state.wall_last, w, state.quan,
        state.discards, state.melds, state.meld_cnt,
        vmap_method="sequential")
    # -1 for an illegal (non->=8-fan) Hu is PRESERVED so trainers can forfeit it
    # rather than pay a payout; only genuine non-Hu terminals are zeroed.
    fan = jnp.where(is_win, fan, 0)
    return state._replace(fan=fan)


# ================================================================ replay tool
def wall_to_codes(wall_strs):
    return np.array([TILE2IDX[t] for t in wall_strs], dtype=np.int8)


def _parse_response(resp, phase, seat, live_tile):
    """Recorded Botzone response string -> (primary 235-code, follow-discard idx)
    for `seat` at the given phase. live_tile = current live discard idx (or -1)."""
    p = (resp or "PASS").split()
    if not p or p[0] == "PASS":
        return A_PASS, 0
    a = p[0]
    if a == "HU":
        return A_HU, 0
    if a == "PLAY":
        return A_PLAY + TILE2IDX[p[1]], 0
    if a == "GANG":
        if len(p) >= 2:                        # concealed AnGang after own draw
            return A_ANGANG + TILE2IDX[p[1]], 0
        return A_GANG + (live_tile if live_tile >= 0 else 0), 0   # melded kong
    if a == "BUGANG":
        return A_BUGANG + TILE2IDX[p[1]], 0
    if a == "PENG":
        follow = TILE2IDX[p[1]] if len(p) >= 2 else 0
        return A_PENG + (live_tile if live_tile >= 0 else 0), follow
    if a == "CHI":
        mid = p[1]
        follow = TILE2IDX[p[2]] if len(p) >= 3 else 0
        midnum = int(mid[1])
        suit = "WTB".index(mid[0])
        claimed_num = live_tile % 9 + 1
        off = claimed_num - midnum + 1
        code = A_CHI + (suit * 7 + (midnum - 2)) * 3 + off
        return code, follow
    return A_PASS, 0


def replay_game(rec):
    """Drive the JAX env with the RECORDED per-seat responses and check the
    event trajectory matches the oracle display stream (action/player/tile/
    tileCHI) and the terminal expected (ending/winner/fan/scores).

    v1 validates FLOW exactly and scores fan/score via the host PyMahjongGB
    (host_score). Returns (ok:bool, msg:str)."""
    quan = int(rec["quan"])
    turns = rec["turns"]
    if turns[0]["display"].get("quan") != quan:
        return False, "INIT quan mismatch"

    # reset already performed seat 0's first DRAW == oracle turns[2].
    st = jax.device_get(reset(jnp.asarray(wall_to_codes(rec["wall"])), quan))
    ok, msg = _cmp_event(st, turns[2]["display"])
    if not ok:
        return False, "turn 2: " + msg

    idx = 2
    for _ in range(len(turns) + 5):
        cur = turns[idx]
        st = jax.device_get(_feed(st, cur["display"], cur.get("responses") or {}))
        idx += 1
        if idx >= len(turns):
            break
        ok, msg = _cmp_event(st, turns[idx]["display"])
        if not ok:
            return False, "turn %d: %s" % (idx, msg)
        if int(st.done):
            break
    # terminal
    exp = rec["expected"]
    if int(st.done) == 0 and exp["ending"] != "draw":
        return False, "env not terminal but oracle is"
    end_map = {1: "zimo", 2: "ron", 3: "draw"}
    got_ending = end_map.get(int(st.ending), "none")
    if got_ending != exp["ending"]:
        return False, "ending %s != %s" % (got_ending, exp["ending"])
    if exp["ending"] == "draw":
        return True, "draw"
    if int(st.winner) != exp["winner"]:
        return False, "winner %d != %d" % (int(st.winner), exp["winner"])
    fan, scores = host_score(st)
    if fan != exp["fan_total"]:
        return False, "fan %d != %d" % (fan, exp["fan_total"])
    if scores != list(exp["scores"]):
        return False, "scores %s != %s" % (scores, exp["scores"])
    return True, "hu fan=%d" % fan


def _feed(st, cur_disp, resp):
    """Given the current event `cur_disp` and its recorded responses, build the
    action vector and step once."""
    live = int(st.last_discard_tile)
    acts = np.zeros((4, 2), dtype=np.int32)
    for s in range(4):
        code, follow = _parse_response(resp.get(str(s)), int(st.phase), s, live)
        acts[s, 0] = code
        acts[s, 1] = follow
    return step(st, jnp.asarray(acts))


def _cmp_event(st, disp):
    """Compare the env's last event (action/player/tile/tileCHI) to an oracle
    display (loose keys)."""
    got_a = EV_NAME.get(int(st.last_ev_a), "?")
    if got_a != disp["action"]:
        return False, "action %s != %s" % (got_a, disp["action"])
    got_p = int(st.last_ev_p)
    exp_p = disp.get("player")
    if exp_p is None:
        exp_p = -1
    if got_p != exp_p:
        return False, "player %d != %s" % (got_p, exp_p)
    got_t = int(st.last_ev_t)
    exp_t = disp.get("tile")
    exp_t = TILE2IDX[exp_t] if exp_t is not None else -1
    if got_t != exp_t:
        gs = TILE_LIST[got_t] if got_t >= 0 else None
        return False, "tile %s != %s" % (gs, disp.get("tile"))
    exp_chi = disp.get("tileCHI")
    if exp_chi is not None:
        if int(st.last_ev_chi) != TILE2IDX[exp_chi]:
            return False, "tileCHI mismatch"
    return True, "ok"
