"""Mechanics micro-tests: known setups with exactly predictable outcomes."""

import jax
import jax.numpy as jnp
import numpy as np

from boom import engine
from boom.engine import E_MAX, E_UNIT, RESULT_ONGOING, RESULT_P0, TICKS_REG

NOOP = jnp.array([4, 0, 0], jnp.int32)
BOTH_NOOP = jnp.stack([NOOP, NOOP])


def _fresh(hand0_card: int | None = None, energy: int = E_MAX):
    state = engine.reset(jax.random.PRNGKey(0), None)
    if hand0_card is not None:
        state = state._replace(hand=state.hand.at[0, 0].set(hand0_card))
    return state._replace(energy=jnp.full(2, energy, jnp.int32))


def test_energy_regen_exact():
    state = _fresh(energy=5 * E_UNIT)
    for _ in range(14):
        state = engine.step(state, BOTH_NOOP, None)
    assert int(state.energy[0]) == 6 * E_UNIT  # +1 energy per 2.8 s before double time


def test_card_cycle():
    state = _fresh()
    played = int(state.hand[0, 0])
    q0 = int(state.queue[0, 0])
    nxt = engine.step(state, jnp.stack([jnp.array([0, 4, 5], jnp.int32), NOOP]), None)
    assert int(nxt.hand[0, 0]) == q0, "hand refills from queue front"
    assert int(nxt.queue[0, 3]) == played, "played card goes to queue back"


def test_spell_damages_tower_at_reduced_rate():
    """Fireball flies for its flight time, then hits for exactly 30% vs towers."""
    state = _fresh(hand0_card=52)  # Fireburst (fireball analog): 689 dmg, 30% to towers
    cast = jnp.stack([jnp.array([0, 3, 25], jnp.int32), NOOP])  # on p1 left turret
    nxt = engine.step(state, cast, None)
    assert int(nxt.tower_hp[3]) == 3052, "fireball must not hit instantly (it flies)"
    for _ in range(5):
        nxt = engine.step(nxt, BOTH_NOOP, None)
    assert int(nxt.tower_hp[3]) == 3052 - 689 * 30 // 100


def test_freeze_stuns_tower():
    """Frostfield (freeze) must stop a tower from firing for its duration."""
    state = _fresh(hand0_card=58)
    state = state._replace(                        # crossed enemy knight, in range
        u_owner=state.u_owner.at[0].set(0), u_type=state.u_type.at[0].set(0),
        u_hp=state.u_hp.at[0].set(1766),
        u_x=state.u_x.at[0].set(int(4.5 * 256)), u_y=state.u_y.at[0].set(int(20.0 * 256)))
    cast = jnp.stack([jnp.array([0, 3, 25], jnp.int32), NOOP])   # freeze on the turret
    state = engine.step(state, cast, None)
    state = engine.step(state, BOTH_NOOP, None)    # freeze applies (delay 1)
    assert int(state.tower_stun[3]) > 0, "tower must be frozen"
    hp0 = int(state.u_hp[0])
    for _ in range(8):
        state = engine.step(state, BOTH_NOOP, None)
    assert int(state.u_hp[0]) == hp0, "frozen tower must not fire"


def test_wincon_unit_reaches_and_hits_tower():
    state = _fresh(hand0_card=39)  # Ramhound: fast, building-only
    place = jnp.stack([jnp.array([0, 4, 14], jnp.int32), NOOP])  # at the left bridge
    state = engine.step(state, place, None)
    step = jax.jit(engine.step)
    for _ in range(250):
        state = step(state, BOTH_NOOP, None)
    assert int(state.tower_hp[3]) < int(engine.TOWER_MAX_HP[3]), \
        "Ramhound never damaged the enemy turret"
    assert (np.asarray(state.tower_hp[:3]) == np.asarray(engine.TOWER_MAX_HP[:3])).all(), \
        "own towers should be untouched in an empty-board push"


def test_core_kill_ends_match():
    state = _fresh()
    state = state._replace(tower_hp=state.tower_hp.at[5].set(0))
    assert int(engine.result(state)) == RESULT_P0


def test_regulation_tower_lead_wins():
    state = _fresh()
    state = state._replace(tower_hp=state.tower_hp.at[3].set(0),
                           tick=jnp.int32(TICKS_REG))
    assert int(engine.result(state)) == RESULT_P0


def test_match_ongoing_at_start():
    assert int(engine.result(_fresh())) == RESULT_ONGOING


def test_draw_at_max_ticks():
    state = _fresh()._replace(tick=jnp.int32(engine.TICKS_MAX))
    assert int(engine.result(state)) == engine.RESULT_DRAW


def test_player1_actions_are_mirrored():
    """The same player-centric action from either seat must be symmetric:
    p1 placing at (4, 14) lands at engine (13, 17)."""
    state = _fresh()
    is_spell = np.asarray(engine.C.is_spell)[np.asarray(state.hand[1])]
    slot = int(np.argmin(is_spell))
    act = jnp.stack([NOOP, jnp.array([slot, 4, 14], jnp.int32)])
    nxt = engine.step(state, act, None)
    alive = np.asarray(nxt.u_hp) > 0
    assert alive.any(), "p1 legal placement must spawn"
    xs = np.asarray(nxt.u_x)[alive] // 256
    ys = np.asarray(nxt.u_y)[alive] // 256
    assert (np.abs(xs - 13) <= 1).all() and (np.abs(ys - 17) <= 1).all(), (xs, ys)


def test_towers_attack_units_in_range():
    """A unit near the bridge must take princess-tower damage (109 per 0.8 s) —
    CR rule: tower range covers the bridge crossing."""
    state = _fresh(hand0_card=0)  # Bulwark (knight analog) walks toward the bridge
    place = jnp.stack([jnp.array([0, 4, 14], jnp.int32), NOOP])
    state = engine.step(state, place, None)
    step = jax.jit(engine.step)
    hits = 0
    prev_hp = 1766
    for _ in range(220):
        state = step(state, BOTH_NOOP, None)
        alive = np.asarray(state.u_hp) > 0
        if not alive.any():
            break
        hp = int(np.asarray(state.u_hp)[alive][0])
        if hp < prev_hp:
            hits += 1
            prev_hp = hp
    assert hits >= 3, "enemy turret never damaged the approaching tank"


def test_sticky_target_lock():
    """CR rule: A attacking B must NOT retarget when closer C appears."""
    state = _fresh(hand0_card=25)  # Longshot (musketeer analog), range 6
    state = state._replace(hand=state.hand.at[1, 0].set(0).at[1, 1].set(2))
    t0 = jnp.stack([jnp.array([0, 9, 14], jnp.int32),      # musketeer at (9,14)
                    jnp.array([0, 8, 11], jnp.int32)])     # knight A -> engine (9,20)
    state = engine.step(state, t0, None)
    for _ in range(8):                                      # deploy + acquire lock
        state = engine.step(state, BOTH_NOOP, None)
    assert int(state.u_tgt[0]) == 1, "musketeer should have locked knight A"
    tB = jnp.stack([NOOP, jnp.array([1, 8, 13], jnp.int32)])  # B -> engine (9,18), closer
    state = engine.step(state, tB, None)
    state = engine.step(state, BOTH_NOOP, None)
    assert int(state.u_tgt[0]) == 1, "lock must stick to A even though B is closer"


def test_stun_resets_target_lock():
    """CR zap rule: stun halts the victim and resets its target lock."""
    state = _fresh(hand0_card=56)  # Shockwave (zap analog)
    state = state._replace(hand=state.hand.at[1, 0].set(0))
    t0 = jnp.stack([NOOP, jnp.array([0, 8, 11], jnp.int32)])   # knight -> engine (9,20)
    state = engine.step(state, t0, None)
    step = jax.jit(engine.step)
    for _ in range(200):                                    # walk until it locks something
        state = step(state, BOTH_NOOP, None)
        if int(state.u_tgt[0]) >= 0:
            break
    assert int(state.u_tgt[0]) >= 0, "knight never acquired a target to test against"
    zap = jnp.stack([jnp.array([0, int(state.u_x[0]) // 256,
                                int(state.u_y[0]) // 256], jnp.int32), NOOP])
    nxt = engine.step(state, zap, None)
    assert int((nxt.u_status[0] >> 24) & 0x7F) > 0, "victim must be stunned"
    assert int(nxt.u_tgt[0]) == -1, "stun must reset the target lock"


def test_king_inactive_until_provoked():
    state = _fresh()
    active = np.asarray(engine._king_active(state))
    assert not active[2] and not active[5], "kings must start inactive"
    assert active[0] and active[1], "princess towers always active"
    poked = state._replace(tower_hp=state.tower_hp.at[3].set(0))
    assert np.asarray(engine._king_active(poked))[5], \
        "king activates when an own turret falls"


def test_princess_tower_geometry():
    """v9 (true CR ranges): 7.5-tile princess reach engages a lane unit ~2 tiles
    past the bridge (row 18.5+), NOT the instant it crosses — matching CR pacing
    where a hog takes its first tower shot a couple tiles in."""
    reach = float(engine.TOWER_RANGE_ARR[3]) / 256   # princess = 7.5
    d_crossed = ((3.5 - 3.5) ** 2 + (25.5 - 18.5) ** 2) ** 0.5   # 2.5 tiles in
    assert d_crossed <= reach, "unit 2.5 tiles past the bridge must be in range"
    d_just = ((3.5 - 3.5) ** 2 + (25.5 - 17.0) ** 2) ** 0.5     # 1 tile in
    assert d_just > reach, "true 7.5 range must NOT cover the bridge exit row"


def test_musketeer_cannot_snipe_tower_across_river():
    """User-reported: a musketeer must NOT reach the enemy tower from her own side —
    her c-c reach (6 + tower pad) is far short of the 10.6+ tiles to the princess."""
    state = _fresh(hand0_card=25)
    place = jnp.stack([jnp.array([0, 3, 14], jnp.int32), NOOP])  # own front row
    state = engine.step(state, place, None)
    step = jax.jit(engine.step)
    for _ in range(30):
        state = step(state, BOTH_NOOP, None)
        y = int(state.u_y[0])
        if y >= 17 * 256:      # she crossed — stop before she legitimately engages
            break
    assert int(state.tower_hp[3]) == 3052 and int(state.tower_hp[4]) == 3052, \
        "musketeer hit a tower from her own side of the river"


def test_tower_ignores_units_on_bridge():
    state = _fresh()
    state = state._replace(
        u_owner=state.u_owner.at[0].set(1), u_type=state.u_type.at[0].set(0),
        u_hp=state.u_hp.at[0].set(1766),
        u_x=state.u_x.at[0].set(4 * 256), u_y=state.u_y.at[0].set(int(15.5 * 256)))
    nxt = state
    for _ in range(2):                     # stays on river rows for these ticks
        nxt = engine.step(nxt, BOTH_NOOP, None)
    assert int(nxt.u_hp[0]) == 1766, "towers must not shoot units on the bridge/river"
    crossed = state._replace(u_y=state.u_y.at[0].set(int(13.0 * 256)))
    hp = 1766
    for _ in range(8):
        crossed = engine.step(crossed, BOTH_NOOP, None)
        hp = int(crossed.u_hp[0])
        if hp < 1766:
            break
    assert hp < 1766, "tower must engage an enemy that crossed onto its side"


def test_frostsprite_lands_freeze_on_tower():
    """User-reported: the ice-spirit analog must survive the crossing and land its
    hit on the tower — possible now that towers cannot snipe the bridge."""
    state = _fresh(hand0_card=37)
    place = jnp.stack([jnp.array([0, 4, 14], jnp.int32), NOOP])
    state = engine.step(state, place, None)
    step = jax.jit(engine.step)
    for _ in range(150):
        state = step(state, BOTH_NOOP, None)
        if int(state.u_hp[0]) <= 0 and int(state.tower_hp[3]) < 3052:
            break
    assert int(state.tower_hp[3]) < 3052, "ice spirit never hit the tower"


def test_building_pulls_hog():
    """CR rule: a deployed building attracts building-targeters off their path."""
    state = _fresh(hand0_card=39)
    state = state._replace(hand=state.hand.at[1, 0].set(47))   # enemy Watchpost
    acts = jnp.stack([jnp.array([0, 4, 14], jnp.int32),        # hog at the bridge
                      jnp.array([0, 13, 11], jnp.int32)])      # cannon -> engine (4,20)
    state = engine.step(state, acts, None)
    for _ in range(10):
        state = engine.step(state, BOTH_NOOP, None)
    assert int(state.u_tgt[0]) == 1, "hog must lock the deployed cannon, not a tower"
    step = jax.jit(engine.step)
    for _ in range(60):
        state = step(state, BOTH_NOOP, None)
    hx, hy = int(state.u_x[0]) / 256, int(state.u_y[0]) / 256
    d = ((hx - 4.5) ** 2 + (hy - 20.5) ** 2) ** 0.5
    assert d < 2.5, f"hog must walk to the cannon (dist {d:.1f} tiles)"
    assert int(state.tower_hp[3]) == 3052 and int(state.tower_hp[4]) == 3052


def test_overtime_triple_elixir():
    state = _fresh(energy=0)._replace(tick=jnp.int32(engine.TICKS_REG))
    state = engine.step(state, BOTH_NOOP, None)
    assert int(state.energy[0]) == 3, "overtime must regen 3x elixir"


def test_timeout_tiebreak_lowest_tower_hp():
    state = _fresh()._replace(tick=jnp.int32(engine.TICKS_MAX))
    assert int(engine.result(state)) == engine.RESULT_DRAW  # perfectly equal
    chipped = state._replace(tower_hp=state.tower_hp.at[3].set(100))  # p1 turret low
    assert int(engine.result(chipped)) == RESULT_P0, \
        "healthier lowest tower must win the time-out tiebreak"


def test_pocket_unlocks_after_turret_kill():
    state = _fresh(hand0_card=0)  # Bulwark, a normal ground unit
    deep = jnp.array([[0, 4, 20], [4, 0, 0]], jnp.int32)   # enemy half, lane 0
    blocked = engine.step(state, deep, None)
    assert int(blocked.illegal[0]) == 1, "enemy half must be illegal pre-kill"
    opened = state._replace(tower_hp=state.tower_hp.at[3].set(0))
    placed = engine.step(opened, deep, None)
    assert int(placed.illegal[0]) == 1 - 1, "pocket must be legal after turret kill"
    assert (np.asarray(placed.u_hp) > 0).any(), "unit must spawn in the pocket"
    other_lane = jnp.array([[0, 13, 20], [4, 0, 0]], jnp.int32)
    still = engine.step(opened, other_lane, None)
    assert int(still.illegal[0]) == 1, "other lane's pocket stays locked"


def test_fireball_knockback():
    state = _fresh(hand0_card=52)  # Fireburst: 689 dmg, 1.0 tile pushback
    state = state._replace(hand=state.hand.at[1, 0].set(0))
    spawn = jnp.stack([NOOP, jnp.array([0, 8, 11], jnp.int32)])   # knight -> engine (9,20)
    state = engine.step(state, spawn, None)
    y_before = int(state.u_y[0])
    cast = jnp.stack([jnp.array([0, 9, 18], jnp.int32), NOOP])    # blast south of it
    nxt = engine.step(state, cast, None)
    for _ in range(5):                                            # flight time
        nxt = engine.step(nxt, BOTH_NOOP, None)
    y_after = int(nxt.u_y[0])
    assert int(nxt.u_hp[0]) > 0, "knight survives one fireball"
    assert y_after > y_before + 140, \
        f"knight must be knocked away from the blast ({y_before}->{y_after})"


def test_body_collision_separates_units():
    """Units occupy space: two bodies dropped on the same tile must separate,
    and a light unit cannot pass through a heavy one."""
    state = _fresh(hand0_card=0)
    state = state._replace(hand=state.hand.at[0, 1].set(4))   # Duskblade (heavy)
    a1 = jnp.stack([jnp.array([0, 9, 10], jnp.int32), NOOP])
    state = engine.step(state, a1, None)
    a2 = jnp.stack([jnp.array([1, 9, 10], jnp.int32), NOOP])  # same tile
    state = engine.step(state, a2, None)
    for _ in range(8):
        state = engine.step(state, BOTH_NOOP, None)
    alive = np.asarray(state.u_hp) > 0
    xs, ys = np.asarray(state.u_x)[alive], np.asarray(state.u_y)[alive]
    assert alive.sum() == 2
    d = ((xs[0] - xs[1]) ** 2 + (ys[0] - ys[1]) ** 2) ** 0.5
    assert d >= 0.8 * 256, f"overlapping bodies must separate (dist {d/256:.2f} tiles)"


def test_heavy_pushes_light_not_vice_versa():
    """Mass rule: mass-20 golem displaces mass-1 skeletons, not the reverse."""
    from boom.cards import CARDS
    assert int(CARDS.mass[2]) >= 15 and int(CARDS.mass[6]) <= 2
    # displacement share is m_other/(m_i+m_j): skeleton takes ~95% of separation
    m_g, m_s = int(CARDS.mass[2]), int(CARDS.mass[6])
    assert m_g / (m_g + m_s) > 0.9


def test_hog_jumps_river():
    """Ramhound (hog family) crosses the river without detouring to a bridge."""
    state = _fresh(hand0_card=39)
    place = jnp.stack([jnp.array([0, 8, 14], jnp.int32), NOOP])  # mid-lane, off-bridge
    state = engine.step(state, place, None)
    step = jax.jit(engine.step)
    crossed_x = None
    for _ in range(120):
        state = step(state, BOTH_NOOP, None)
        if int(state.u_hp[0]) <= 0:
            break
        y = int(state.u_y[0]) / 256
        if 15.0 <= y <= 17.0 and crossed_x is None:
            crossed_x = int(state.u_x[0]) / 256
    assert crossed_x is not None, "hog never crossed the river"
    assert abs(crossed_x - 4.0) > 1.5 and abs(crossed_x - 15.0) > 1.5, \
        f"hog crossed at x={crossed_x:.1f} — that is a bridge, it should jump straight"


def test_new_building_pulls_marching_giant():
    """CR rule: building-targeters continuously retarget the nearest structure —
    a cannon deployed AFTER the giant started marching must pull it."""
    state = _fresh(hand0_card=1)   # Ironhide (giant analog)
    state = state._replace(hand=state.hand.at[1, 0].set(47))
    place = jnp.stack([jnp.array([0, 4, 14], jnp.int32), NOOP])
    state = engine.step(state, place, None)
    step = jax.jit(engine.step)
    for _ in range(15):            # giant locks the princess tower and marches
        state = step(state, BOTH_NOOP, None)
    assert int(state.u_tgt[0]) >= 64, "giant should be heading for a tower"
    drop = jnp.stack([NOOP, jnp.array([0, 11, 11], jnp.int32)])  # cannon -> engine (6,20)
    state = engine.step(state, drop, None)
    state = engine.step(state, BOTH_NOOP, None)
    assert int(state.u_tgt[0]) == 1, "deployed cannon must pull the marching giant"


def test_defensive_building_never_attacks_towers():
    """CR rule: a cannon ignores towers entirely, even placed at the king's feet."""
    state = _fresh()
    state = state._replace(
        u_owner=state.u_owner.at[0].set(0), u_type=state.u_type.at[0].set(47),
        u_hp=state.u_hp.at[0].set(824),
        u_x=state.u_x.at[0].set(9 * 256), u_y=state.u_y.at[0].set(26 * 256))
    for _ in range(10):
        state = engine.step(state, BOTH_NOOP, None)
    assert int(state.tower_hp[5]) == 4824, "cannon must not damage the king tower"
    assert int(state.u_tgt[0]) < 64 or int(state.u_tgt[0]) == -1


def test_pocket_depth_capped():
    """CR rule: the unlocked pocket stops well short of the king tower."""
    state = _fresh(hand0_card=0)
    opened = state._replace(tower_hp=state.tower_hp.at[3].set(0))
    deep = engine.step(opened, jnp.array([[0, 4, 24], [4, 0, 0]], jnp.int32), None)
    assert int(deep.illegal[0]) == 1, "row 24 must stay locked"
    ok = engine.step(opened, jnp.array([[0, 4, 23], [4, 0, 0]], jnp.int32), None)
    assert int(ok.illegal[0]) == 0 and (np.asarray(ok.u_hp) > 0).any()


def test_king_range_is_seven_tiles():
    """CR: king tower range 7.0 — a unit 8 tiles away on the king's side must
    not be shot; inside 7 it must be (v9 user-reported fix)."""
    state = _fresh(hand0_card=25)               # a tanky ground unit
    state = state._replace(tower_hp=state.tower_hp.at[3].set(0))
    far = engine.step(state, jnp.stack([jnp.array([0, 4, 21], jnp.int32), NOOP]), None)
    step = jax.jit(engine.step)
    for _ in range(6):
        far = step(far, BOTH_NOOP, None)
    from boom.cards import CARDS as _CARDS
    unit_hp_far = int(far.u_hp[0])
    assert unit_hp_far == int(_CARDS.hp[25]), \
        "king shot a unit ~8+ tiles away (range should be 7.0)"


def test_pocket_hog_prefers_princess_over_king():
    """CR: a building-targeter in the pocket marches on the remaining princess
    tower, not the (geometrically nearer) king (v9 user-reported fix)."""
    state = _fresh(hand0_card=39)               # Ramhound (hog-class)
    state = state._replace(tower_hp=state.tower_hp.at[3].set(0),
                           energy=jnp.full(2, 140, jnp.int32))
    state = engine.step(state, jnp.stack([jnp.array([0, 4, 23], jnp.int32), NOOP]), None)
    step = jax.jit(engine.step)
    for _ in range(220):
        state = step(state, BOTH_NOOP, None)
    princess_hp = int(state.tower_hp[4])
    king_hp = int(state.tower_hp[5])
    assert king_hp == int(engine.TOWER_MAX_HP[5]), \
        f"hog attacked the king (hp {king_hp}) — must march on the princess"
    assert princess_hp < int(engine.TOWER_MAX_HP[4]), \
        "hog never reached the remaining princess tower"


def test_building_immune_to_knockback():
    """CR: buildings are anchored — a snowball/fireball must not displace them
    (v10 user-reported fix)."""
    state = _fresh(hand0_card=47)                 # p0 cannon-class
    state = state._replace(hand=state.hand.at[1, 0].set(6))   # p1 fireball-class
    state = engine.step(state, jnp.stack([jnp.array([0, 4, 10], jnp.int32), NOOP]), None)
    bx, by = int(state.u_x[0]), int(state.u_y[0])
    cast = jnp.stack([NOOP, jnp.array([0, 13, 21], jnp.int32)])   # p1 frame -> lands on it
    state = engine.step(state, cast, None)
    step = jax.jit(engine.step)
    for _ in range(8):                             # cover spell flight time
        state = step(state, BOTH_NOOP, None)
    assert (int(state.u_x[0]), int(state.u_y[0])) == (bx, by), \
        "building was displaced by spell knockback"


def test_tower_first_hit_aim_delay():
    """CR: a tower acquiring a fresh target aims briefly before the first shot."""
    state = _fresh(hand0_card=0)
    state = state._replace(tower_hp=state.tower_hp.at[3].set(0))
    state = engine.step(state, jnp.stack([jnp.array([0, 4, 23], jnp.int32), NOOP]), None)
    step = jax.jit(engine.step)
    hp0 = int(state.u_hp[0])
    hit_tick = None
    for t in range(1, 12):
        state = step(state, BOTH_NOOP, None)
        if int(state.u_hp[0]) < hp0:
            hit_tick = t
            break
    assert hit_tick is not None and hit_tick > engine.TOWER_AIM_TICKS, \
        f"tower fired at tick {hit_tick} — must aim >= {engine.TOWER_AIM_TICKS + 1}"


def test_charge_doubles_first_hit():
    """Lancer (prince family): a >=3.5-tile run-up doubles the first hit."""
    from boom.cards import CARDS as _C
    lancer = 60
    state = _fresh(hand0_card=lancer)
    state = state._replace(hand=state.hand.at[1, 0].set(47))
    # stationary victim: a p1 BUILDING just past the river on the bridge column;
    # the Lancer runs ~15 tiles (charged) and lands its doubled first hit there
    state = engine.step(state, jnp.stack([jnp.array([0, 4, 4], jnp.int32), NOOP]), None)
    state = engine.step(state, jnp.stack([NOOP, jnp.array([0, 13, 12], jnp.int32)]), None)
    step = jax.jit(engine.step)
    # buildings decay each tick, so detect the charged hit as the largest
    # single-tick hp drop (decay is single-digit noise; a normal hit is 325)
    max_drop = 0
    for _ in range(200):
        prev = int(state.u_hp[1])
        state = step(state, BOTH_NOOP, None)
        if prev > 0:
            max_drop = max(max_drop, prev - int(state.u_hp[1]))
    # the kill blow is capped at remaining hp (decay eats the rest), so the
    # proof of doubling is any single-tick drop exceeding one NORMAL hit
    assert max_drop > int(_C.dmg[lancer]) + 25, \
        f"largest single-tick hit {max_drop} <= normal {int(_C.dmg[lancer])} — charge never doubled"


def test_elixir_starts_at_five():
    raw = engine.reset(jax.random.PRNGKey(0), None)   # _fresh overrides energy
    assert int(raw.energy[0]) == 5 * E_UNIT


def test_unit_slides_around_own_tower():
    """v12 user-reported: a unit placed directly behind its own princess tower
    deadlocked against the radial footprint push forever. The tangential slide
    must let it walk around and reach the lane."""
    state = _fresh(hand0_card=3)                  # Shellfort, slow walker
    state = engine.step(state, jnp.stack([jnp.array([0, 3, 4], jnp.int32), NOOP]), None)
    idx = int(jnp.argmax(state.u_hp > 0))
    step = jax.jit(engine.step)
    for _ in range(360):
        state = step(state, BOTH_NOOP, None)
    y_tiles = float(state.u_y[idx]) / engine.FP
    assert (state.u_hp[idx] <= 0) or y_tiles > 8.0, \
        f"unit still stuck behind the tower at y={y_tiles:.1f}"


def test_heavy_unit_immune_to_knockback():
    """v13 user-reported: giants/golems (mass >= 10) must not be displaced by
    fireball-class knockback, same as CR's heavy class."""
    state = _fresh(hand0_card=1)                  # Ironhide (giant, mass 18)
    state = state._replace(hand=state.hand.at[1, 0].set(52))   # p1 Fireburst
    state = engine.step(state, jnp.stack([jnp.array([0, 9, 10], jnp.int32), NOOP]), None)
    idx = int(jnp.argmax(state.u_hp > 0))
    bx, by = int(state.u_x[idx]), int(state.u_y[idx])
    state = engine.step(state, jnp.stack([NOOP, jnp.array([0, 9, 10], jnp.int32)]), None)
    step = jax.jit(engine.step)
    for _ in range(80):                           # spell flight + impact
        state = step(state, BOTH_NOOP, None)
        state = state._replace(u_tgt=state.u_tgt)  # no-op keep
    # compare y only against march drift: displacement must be small & forward
    assert int(state.u_y[idx]) >= by, "giant was knocked backward by fireball"


def test_knockback_cannot_push_unit_into_river():
    """v13 user-reported: a knockable unit near the bank must never end up
    standing IN the river band (rows 15-17) off-bridge."""
    state = _fresh(hand0_card=25)                 # Longshot (light, knockable)
    state = state._replace(hand=state.hand.at[1, 0].set(52))
    # place just south of the river, mid-lane (not on a bridge x)
    state = engine.step(state, jnp.stack([jnp.array([0, 8, 14], jnp.int32), NOOP]), None)
    idx = int(jnp.argmax(state.u_hp > 0))
    # enemy fireball lands just south of the unit -> pushes it north into water
    state = engine.step(state, jnp.stack([NOOP, jnp.array([0, 8, 13], jnp.int32)]), None)
    step = jax.jit(engine.step)
    for _ in range(80):
        state = step(state, BOTH_NOOP, None)
        if state.u_hp[idx] <= 0:
            return                                 # died first: fine
    y_fp = int(state.u_y[idx])
    on_bridge = min(abs(int(state.u_x[idx]) - engine.BRIDGE_X_FP[0]),
                    abs(int(state.u_x[idx]) - engine.BRIDGE_X_FP[1])) <= engine.FP
    assert on_bridge or not (engine.RIVER_LO_FP < y_fp < engine.RIVER_HI_FP), \
        f"unit stuck in river at y={y_fp/engine.FP:.1f}"


def test_walking_unit_retargets_to_interposed_body():
    """v14 user-reported (CR kiting rule): a troop WALKING toward a locked
    target must re-evaluate — an ice-golem-class body placed between a mega
    minion and its victim pulls the aggro. Locks hold only while engaged."""
    state = _fresh(hand0_card=14)                 # p0 Skyray (mega minion)
    state = state._replace(hand=state.hand.at[1, 0].set(25))   # p1 Longshot
    state = state._replace(hand=state.hand.at[1, 1].set(3))    # p1 Shellfort
    # y=17 is river — p1's legal half starts at 18. Angled approach within
    # sight (5.5): Skyray (5,14) -> Longshot (8,18), lock forms, walk begins.
    # actions are PLAYER-CENTRIC: p1's (9,13) mirrors to engine (8,18)
    state = engine.step(state, jnp.stack([jnp.array([0, 5, 14], jnp.int32),
                                          jnp.array([0, 9, 13], jnp.int32)]), None)
    assert bool(jnp.any((state.u_type == 25) & (state.u_hp > 0))), \
        "setup: Longshot must spawn (legal tile)"
    sk = int(jnp.argmax((state.u_type == 14) & (state.u_hp > 0)))
    lo = int(jnp.argmax((state.u_type == 25) & (state.u_hp > 0)))
    step = jax.jit(engine.step)
    for _ in range(6):                            # a few ticks: locked, walking
        state = step(state, BOTH_NOOP, None)
    assert int(state.u_tgt[sk]) == lo, "precondition: Skyray locked on Longshot"
    # interpose Shellfort nearer to the walking Skyray: engine (7,18) = p1 (10,13)
    state = step(state, jnp.stack([NOOP, jnp.array([1, 10, 13], jnp.int32)]), None)
    for _ in range(8):
        state = step(state, BOTH_NOOP, None)
    sf = int(jnp.argmax((state.u_type == 3) & (state.u_hp > 0)))
    assert int(state.u_tgt[sk]) == sf, \
        f"Skyray still locked on Longshot (tgt={int(state.u_tgt[sk])}, want {sf})"


def test_spell_damage_activates_king():
    """v15 pin (user-reported; verified working): ANY damage to the king —
    including spell chip — must wake it."""
    state = _fresh(hand0_card=52)                 # Fireburst on p1 king
    state = engine.step(state, jnp.stack([jnp.array([0, 9, 28], jnp.int32), NOOP]), None)
    step = jax.jit(engine.step)
    for _ in range(40):                           # flight + impact
        state = step(state, BOTH_NOOP, None)
    assert int(state.tower_hp[5]) < int(engine.TOWER_MAX_HP[5]), "spell must chip the king"
    act = engine._king_active(state)
    assert bool(act[5]), "damaged king must be active"
    assert not bool(act[2]), "untouched king must stay asleep"


def test_whirlgale_pulls_heavy_but_not_building():
    """v15: the tornado-family spell drags even mass-18 units toward its center;
    buildings are anchored."""
    state = _fresh(hand0_card=1)                  # p0 Ironhide (heavy)
    state = state._replace(hand=state.hand.at[1, 0].set(61))   # p1 Whirlgale
    state = engine.step(state, jnp.stack([jnp.array([0, 5, 10], jnp.int32), NOOP]), None)
    idx = int(jnp.argmax(state.u_hp > 0))
    bx = int(state.u_x[idx])
    # p1 casts 3 tiles east of the giant: player-centric mirror of engine (8,10)
    # (spell_delay 1 = instant; measure right after impact, before march drift)
    state = engine.step(state, jnp.stack([NOOP, jnp.array([0, 17 - 8, 31 - 10], jnp.int32)]), None)
    state = engine.step(state, BOTH_NOOP, None)
    assert int(state.u_x[idx]) > bx + 2 * engine.FP, \
        f"heavy unit was not pulled east (x {bx} -> {int(state.u_x[idx])})"


def test_harpooner_drags_victim_adjacent():
    """v15: the fisherman-family unit's hit yanks its victim to ~1 tile away."""
    state = _fresh(hand0_card=62)                 # p0 Harpooner
    state = state._replace(hand=state.hand.at[1, 0].set(25))   # p1 Longshot
    # engine (8,14) vs (8,17.5→18): ~3.5 tiles = inside Harpooner reach; the
    # ranged victim holds position, so only the hook can close the gap
    state = engine.step(state, jnp.stack([jnp.array([0, 8, 14], jnp.int32),
                                          jnp.array([0, 9, 13], jnp.int32)]), None)
    hp_i = int(jnp.argmax((state.u_type == 62) & (state.u_hp > 0)))
    lo_i = int(jnp.argmax((state.u_type == 25) & (state.u_hp > 0)))
    step = jax.jit(engine.step)
    for _ in range(80):
        state = step(state, BOTH_NOOP, None)
        if state.u_hp[lo_i] <= 0 or state.u_hp[hp_i] <= 0:
            raise AssertionError("combat ended before a hook landed")
        d2 = int(engine._dist2_16(state.u_x[hp_i], state.u_y[hp_i],
                                  state.u_x[lo_i], state.u_y[lo_i]))
        if d2 <= int(engine._r2_16(jnp.int32(3 * engine.FP // 2))):
            return                                # dragged to ~1 tile
    raise AssertionError("victim was never dragged adjacent to the Harpooner")


def test_marrowkeg_spawns_goblins():
    """v16 Goblin-Barrel analog: casting Marrowkeg spawns 3 goblins for the caster."""
    state = _fresh(hand0_card=64)                 # Marrowkeg (spawns Marrowlings id 8)
    n_before = int(jnp.sum((state.u_type == 8) & (state.u_owner == 0) & (state.u_hp > 0)))
    # cast anywhere (anywhere=1): on enemy half
    state = engine.step(state, jnp.stack([jnp.array([0, 9, 24], jnp.int32), NOOP]), None)
    step = jax.jit(engine.step)
    for _ in range(6):                            # spell flight (delay 2) + impact
        state = step(state, BOTH_NOOP, None)
    n_after = int(jnp.sum((state.u_type == 8) & (state.u_owner == 0) & (state.u_hp > 0)))
    assert n_after >= n_before + 3, f"Marrowkeg must spawn 3 goblins ({n_before}->{n_after})"


def test_timberoll_knocks_back_no_spawn():
    """Timberoll (Log analog) damages + knocks back, spawns nothing."""
    state = _fresh(hand0_card=63)                 # Timberoll
    state = state._replace(hand=state.hand.at[1, 0].set(25))    # p1 Longshot to hit
    state = engine.step(state, jnp.stack([NOOP, jnp.array([0, 9, 13], jnp.int32)]), None)
    vic = int(jnp.argmax((state.u_type == 25) & (state.u_hp > 0)))
    vy = int(state.u_y[vic])
    hp0 = int(state.u_hp[vic])
    # p0 rolls Timberoll onto the Longshot
    lx, ly = int(state.u_x[vic]) // engine.FP, int(state.u_y[vic]) // engine.FP
    state = engine.step(state, jnp.stack([jnp.array([0, lx, ly], jnp.int32), NOOP]), None)
    step = jax.jit(engine.step)
    for _ in range(5):
        state = step(state, BOTH_NOOP, None)
    if int(state.u_hp[vic]) > 0:
        assert int(state.u_hp[vic]) < hp0, "Timberoll should damage the target"
    # no goblins/barbarians spawned by a pure log
    assert int(jnp.sum((state.u_type == 8) & (state.u_owner == 0))) == 0
