"""Property tests: invariants that must hold on random-legal rollouts."""

import jax
import jax.numpy as jnp
import numpy as np

from boom import engine, vec
from boom.cards import CARDS
from boom.engine import E_MAX, MAX_UNITS


def _final_states(batch=16, ticks=600, seed=0):
    states, _ = vec.rollout_random_jit(jax.random.PRNGKey(seed), batch, ticks, None, False)
    return jax.tree_util.tree_map(np.asarray, states)


def test_energy_bounds():
    s = _final_states()
    assert (s.energy >= 0).all() and (s.energy <= E_MAX).all()


def test_no_negative_hp_and_caps():
    s = _final_states()
    assert (s.u_hp >= 0).all(), "negative unit hp"
    assert (s.tower_hp >= 0).all(), "negative tower hp"
    alive = s.u_hp > 0
    max_hp = np.asarray(CARDS.hp)[s.u_type]
    assert (s.u_hp[alive] <= max_hp[alive]).all(), "hp above card max (heal overflow?)"
    assert (alive.sum(axis=-1) <= MAX_UNITS).all()


def test_positions_on_board():
    s = _final_states()
    assert (s.u_x >= 0).all() and (s.u_x < engine.W_FP).all()
    assert (s.u_y >= 0).all() and (s.u_y < engine.H_FP).all()


def test_legal_mask_soundness():
    """Random-LEGAL agents must never trip the illegal-action counter (AGENTS.md §2:
    the mask is sound iff sampling from it never falls back)."""
    s = _final_states(batch=32, ticks=400, seed=1)
    assert (s.illegal == 0).all(), f"legal mask unsound: {s.illegal.sum()} illegal plays"


def test_illegal_action_is_counted_noop():
    """An explicitly illegal placement (unit on enemy half) is a counted no-op."""
    state = engine.reset(jax.random.PRNGKey(0), None)
    is_spell = np.asarray(CARDS.is_spell)[np.asarray(state.hand[0])]
    slot = int(np.argmin(is_spell))          # a unit card slot
    assert is_spell[slot] == 0
    bad = jnp.array([[slot, 4, 25], [4, 0, 0]], jnp.int32)   # y=25 not on own half
    before = state.energy[0]
    nxt = engine.step(state, bad, None)
    assert int(nxt.illegal[0]) == 1
    assert int(nxt.energy[0]) == int(before) + 1              # only regen, no cost paid
    assert (np.asarray(nxt.u_hp) == 0).all(), "no unit spawned from illegal play"


def test_hand_queue_always_partition_deck():
    s = _final_states(batch=8, ticks=500, seed=2)
    for b in range(8):
        for p in range(2):
            cycle = sorted(s.hand[b, p].tolist() + s.queue[b, p].tolist())
            assert len(set(cycle)) == 8, "hand+queue must stay a permutation of the deck"


def test_matches_progress():
    """Random aggressive play must destroy towers within regulation in most games."""
    s = _final_states(batch=32, ticks=900, seed=3)
    towers_lost = (s.tower_hp == 0).sum()
    assert towers_lost > 0, "no tower ever destroyed across 32 random matches"


def test_river_impassable_for_ground_units():
    """CR terrain: ground non-jumping units are never on the river rows outside
    the bridge spans."""
    s = _final_states(batch=16, ticks=500, seed=5)
    air = np.asarray(CARDS.air)[s.u_type] == 1
    jumps = np.asarray(CARDS.jumps)[s.u_type] == 1
    ground = (s.u_hp > 0) & ~air & ~jumps
    in_river = (s.u_y >= 15 * 256) & (s.u_y < 17 * 256)
    on_bridge = (((s.u_x >= 3 * 256) & (s.u_x < 5 * 256))
                 | ((s.u_x >= 14 * 256) & (s.u_x < 16 * 256)))
    bad = ground & in_river & ~on_bridge
    assert not bad.any(), f"{bad.sum()} ground units standing in the water"
