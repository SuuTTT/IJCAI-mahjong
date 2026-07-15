"""Warpath v1 core-loop pins: economy, production, combat, win condition."""
import jax
import jax.numpy as jnp

from warpath import engine

NOOP = jnp.array([0, 0, 0], jnp.int32)
BOTH_NOOP = jnp.stack([NOOP, NOOP])


def _run(state, n, acts=None):
    step = jax.jit(engine.step)
    for _ in range(n):
        state = step(state, acts if acts is not None else BOTH_NOOP)
    return state


def _build_place(state, btype, x, y, credits=9000):
    """RA2 build: start sidebar construction, wait, place at (x,y)."""
    state = state._replace(credits=state.credits.at[0].set(credits))
    state = engine.step(state, jnp.stack([jnp.array([1, btype, 0], jnp.int32), NOOP]))
    t = int(jnp.asarray(engine.B_BUILD_T)[btype])
    state = _run(state, t + 1)
    state = engine.step(state, jnp.stack([jnp.array([6, 0, x * 64 + y], jnp.int32), NOOP]))
    assert bool(jnp.any((state.b_type == btype) & (state.b_owner == 0))), \
        f"building {btype} not placed at ({x},{y})"
    return state


def test_harvester_economy_cycles():
    state = engine.reset()
    state = _build_place(state, 1, 4, 12)          # power
    state = _build_place(state, 8, 10, 15)         # refinery -> free harvester
    c0 = int(state.credits[0])
    state = _run(state, 600)
    assert int(state.credits[0]) > c0 + 200, \
        f"harvester never paid out (credits {c0} -> {int(state.credits[0])})"
    assert int(jnp.sum(state.ore)) < engine.ORE_START * engine.N_ORE, "ore untouched"


def test_train_requires_barracks():
    state = engine.reset()
    train = jnp.stack([jnp.array([2, 1, 0], jnp.int32), NOOP])
    state = engine.step(state, train)
    # no barracks yet -> illegal
    assert int(state.illegal[0]) == 1


def test_prereq_graph_gates_and_unlocks():
    """RA2 rule: buildable = owns producer + ALL prerequisite buildings alive."""
    state = engine.reset()
    state = state._replace(credits=state.credits.at[0].set(9000))
    # barracks before power -> pending rejected (needs power plant)
    state = engine.step(state, jnp.stack([jnp.array([1, 2, 0], jnp.int32), NOOP]))
    assert int(state.pend_type[0]) == -1, "barracks must be locked before power"
    # tree: power -> barracks -> factory
    state = _build_place(state, 1, 4, 12)
    state = _build_place(state, 2, 8, 15)
    state = _build_place(state, 3, 9, 12)
    # bulldog (lab tech) locked without a lab even with a factory
    bad = int(state.illegal[0])
    state = engine.step(state, jnp.stack([jnp.array([2, 4, 0], jnp.int32), NOOP]))
    assert int(state.illegal[0]) == bad + 1, "bulldog must be lab-gated"


def test_build_then_train_then_fight_to_win():
    state = engine.reset()
    state = _build_place(state, 1, 4, 12)
    state = _build_place(state, 2, 8, 15)
    # queue riflemen whenever idle and shove them at the enemy conyard
    step = jax.jit(engine.step)
    atk = jnp.array([3, 0, 34 * 64 + 15], jnp.int32)
    train = jnp.array([2, 1, 0], jnp.int32)
    for t in range(2800):
        act = train if t % 40 == 0 else (atk if t % 40 == 20 else NOOP)
        state = step(state, jnp.stack([act, NOOP]))
        if int(engine.result(state)) == 0:
            break
    assert int(engine.result(state)) == 0, \
        f"rifle flood never won (result {int(engine.result(state))}, " \
        f"enemy con hp {int(state.b_hp[1])})"


def test_per_unit_orders_move_only_selected():
    """cmd 5: a selection order moves THAT unit, not the whole army."""
    state = engine.reset()
    ut, uo, uh = state.u_type, state.u_owner, state.u_hp
    ux, uy = state.u_x, state.u_y
    for i, x in [(4, 10), (5, 12)]:
        ut, uo = ut.at[i].set(1), uo.at[i].set(0)
        uh = uh.at[i].set(engine.U_HP[1])
        ux, uy = ux.at[i].set(x * engine.FP), uy.at[i].set(8 * engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    order = jnp.stack([jnp.array([5, 4, 20 * 64 + 25], jnp.int32), NOOP])
    state = engine.step(state, order)
    step = jax.jit(engine.step)
    for _ in range(40):
        state = step(state, BOTH_NOOP)
    moved = abs(int(state.u_y[4]) - 8 * engine.FP)
    stayed = abs(int(state.u_y[5]) - 8 * engine.FP)
    assert moved > 4 * engine.FP, "ordered unit did not move"
    assert stayed < engine.FP, "unordered unit moved"


def test_rocketeer_beats_tank_cost_for_cost():
    state = engine.reset()
    # hand-spawn: 2 rocketeers (480c) vs 1 tank (500c) mid-field
    ut, uo, uh = state.u_type, state.u_owner, state.u_hp
    ux, uy = state.u_x, state.u_y
    for i, (t, o, x) in enumerate([(2, 0, 18), (2, 0, 19), (3, 1, 22)], start=4):
        ut, uo = ut.at[i].set(t), uo.at[i].set(o)
        uh = uh.at[i].set(engine.U_HP[t])
        ux = ux.at[i].set(x * engine.FP)
        uy = uy.at[i].set(15 * engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    state = _run(state, 300)
    rockets_alive = bool(jnp.any((state.u_type == 2) & (state.u_hp > 0)))
    tank_alive = bool(jnp.any((state.u_type == 3) & (state.u_hp > 0)))
    assert rockets_alive and not tank_alive, \
        f"anti-armor identity broken (rockets {rockets_alive}, tank {tank_alive})"


def test_production_queue_ra2_style():
    """Clicking N times queues N units that build back-to-back (RA2 queue)."""
    state = engine.reset()
    state = _build_place(state, 1, 4, 12)
    state = _build_place(state, 2, 8, 15)
    state = state._replace(credits=state.credits.at[0].set(2000))
    train = jnp.stack([jnp.array([2, 1, 0], jnp.int32), NOOP])
    for _ in range(4):                            # click 4x: 1 starts + 3 queue
        state = engine.step(state, train)
    assert int(jnp.sum(state.prod_q[0, 0] >= 0)) == 3, "queue should hold 3"
    # RA2: only the item that STARTED is charged; the 3 queued are still free
    assert int(state.credits[0]) == 2000 - int(engine.U_COST[1]), \
        "only the started item is charged"
    state = _run(state, 5 * 25 + 20)              # all four build out (charged as they start)
    n_rifles = int(jnp.sum((state.u_type == 1) & (state.u_owner == 0) & (state.u_hp > 0)))
    assert n_rifles == 4, f"expected 4 riflemen, got {n_rifles}"


def test_harvester_obeys_manual_move():
    """RA2: a harvester honors a manual move order, then resumes harvesting."""
    state = engine.reset()
    state = _build_place(state, 1, 4, 12)
    state = _build_place(state, 8, 10, 15)         # refinery -> free harvester (slot varies)
    hi = int(jnp.argmax((state.u_type == 0) & (state.u_owner == 0)))
    hx0 = float(state.u_x[hi])
    order = jnp.stack([jnp.array([5, hi, 25 * 64 + 25], jnp.int32), NOOP])
    state = engine.step(state, order)
    step = jax.jit(engine.step)
    for _ in range(40):
        state = step(state, BOTH_NOOP)
    assert abs(float(state.u_x[hi]) - hx0) > 4 * engine.FP, \
        "harvester ignored the manual move order"


def test_tank_crushes_infantry():
    """RA2 crush: a tank driving over enemy infantry kills it instantly."""
    state = engine.reset()
    ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
    ut, uo = ut.at[4].set(3), uo.at[4].set(0)          # p0 tank
    uh = uh.at[4].set(engine.U_HP[3]); ux = ux.at[4].set(15*engine.FP); uy = uy.at[4].set(15*engine.FP)
    ut, uo = ut.at[5].set(1), uo.at[5].set(1)          # p1 rifleman right next to it
    uh = uh.at[5].set(engine.U_HP[1]); ux = ux.at[5].set(int(15.2*engine.FP)); uy = uy.at[5].set(15*engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    state = engine.step(state, BOTH_NOOP)
    assert int(state.u_hp[5]) <= 0, "tank should crush the adjacent enemy infantry"


def test_vehicle_does_not_crush_own_infantry():
    state = engine.reset()
    ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
    ut, uo = ut.at[4].set(3), uo.at[4].set(0)
    uh = uh.at[4].set(engine.U_HP[3]); ux = ux.at[4].set(15*engine.FP); uy = uy.at[4].set(15*engine.FP)
    ut, uo = ut.at[5].set(1), uo.at[5].set(0)          # FRIENDLY rifleman
    uh = uh.at[5].set(engine.U_HP[1]); ux = ux.at[5].set(int(15.2*engine.FP)); uy = uy.at[5].set(15*engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    state = engine.step(state, BOTH_NOOP)
    assert int(state.u_hp[5]) > 0, "must not crush friendly infantry"


def test_tank_splash_destroys_miner():
    """AoE: a tank firing near a harvester wrecks the harvester via splash."""
    state = engine.reset()
    ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
    ut, uo = ut.at[4].set(3), uo.at[4].set(0)          # p0 tank
    uh = uh.at[4].set(engine.U_HP[3]); ux = ux.at[4].set(10*engine.FP); uy = uy.at[4].set(10*engine.FP)
    # enemy rifleman (primary target) + harvester right beside it
    ut, uo = ut.at[5].set(1), uo.at[5].set(1)
    uh = uh.at[5].set(engine.U_HP[1]); ux = ux.at[5].set(int(13.5*engine.FP)); uy = uy.at[5].set(10*engine.FP)
    ut, uo = ut.at[6].set(0), uo.at[6].set(1)
    uh = uh.at[6].set(engine.U_HP[0]); ux = ux.at[6].set(int(14.0*engine.FP)); uy = uy.at[6].set(10*engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    step = jax.jit(engine.step)
    hp0 = int(state.u_hp[6])
    for _ in range(60):
        state = step(state, BOTH_NOOP)
    assert int(state.u_hp[6]) < hp0, "tank splash should damage the nearby harvester"


def test_defense_turret_shoots_attacker():
    """A gun turret auto-fires at an enemy unit in range."""
    state = engine.reset()
    # power plant (so the turret is powered) + turret near an enemy rifleman
    bt, bo, bh, bx, by = state.b_type, state.b_owner, state.b_hp, state.b_x, state.b_y
    bt, bo = bt.at[3].set(1), bo.at[3].set(0)          # power plant
    bh = bh.at[3].set(engine.B_HP[1]); bx = bx.at[3].set(6*engine.FP); by = by.at[3].set(15*engine.FP)
    bt, bo = bt.at[4].set(6), bo.at[4].set(0)          # gun turret
    bh = bh.at[4].set(engine.B_HP[6]); bx = bx.at[4].set(10*engine.FP); by = by.at[4].set(10*engine.FP)
    state = state._replace(b_type=bt, b_owner=bo, b_hp=bh, b_x=bx, b_y=by)
    ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
    ut, uo = ut.at[5].set(1), uo.at[5].set(1)
    uh = uh.at[5].set(engine.U_HP[1]); ux = ux.at[5].set(13*engine.FP); uy = uy.at[5].set(10*engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    step = jax.jit(engine.step)
    hp0 = int(state.u_hp[5])
    for _ in range(30):
        state = step(state, BOTH_NOOP)
    assert int(state.u_hp[5]) < hp0, "turret should have shot the enemy rifleman"


def test_rocketeer_flies_rifle_cannot_hit():
    """Air layer: a rifleman (ground) can't target a flying rocketeer; a
    rocketeer CAN (it has anti-air)."""
    state = engine.reset()
    ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
    ut, uo = ut.at[4].set(1), uo.at[4].set(0)          # p0 rifleman
    uh = uh.at[4].set(engine.U_HP[1]); ux = ux.at[4].set(10*engine.FP); uy = uy.at[4].set(10*engine.FP)
    ut, uo = ut.at[5].set(2), uo.at[5].set(1)          # p1 rocketeer (air), in rifle range
    uh = uh.at[5].set(engine.U_HP[2]); ux = ux.at[5].set(12*engine.FP); uy = uy.at[5].set(10*engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    step = jax.jit(engine.step)
    hp0 = int(state.u_hp[5])
    for _ in range(40):
        state = step(state, BOTH_NOOP)
    assert int(state.u_hp[5]) == hp0, "rifleman must NOT damage the flying rocketeer"


def test_queue_charges_on_start_not_on_queue():
    """RA2: money is spent when a queued unit BEGINS, not when queued."""
    state = _build_place(state=engine.reset(), btype=1, x=4, y=12)
    state = _build_place(state, 2, 8, 15)
    state = state._replace(credits=state.credits.at[0].set(500))  # 2 rifles = $400
    train = jnp.stack([jnp.array([2, 1, 0], jnp.int32), NOOP])
    state = engine.step(state, train)                  # first starts: -200
    assert int(state.credits[0]) == 300
    for _ in range(3):                                 # queue 3 more: FREE
        state = engine.step(state, train)
    assert int(state.credits[0]) == 300, "queuing must not charge"
    assert int(jnp.sum(state.prod_q[0, 0] >= 0)) == 3


def test_cancel_building_refunds():
    state = engine.reset()
    c0 = int(state.credits[0])
    state = engine.step(state, jnp.stack([jnp.array([1, 1, 0], jnp.int32), NOOP]))  # power, -800
    assert int(state.credits[0]) == c0 - 800
    assert int(state.pend_type[0]) == 1
    state = engine.step(state, jnp.stack([jnp.array([7, 0, 0], jnp.int32), NOOP]))  # cancel
    assert int(state.credits[0]) == c0, "cancel must refund the full cost"
    assert int(state.pend_type[0]) == -1


def test_no_starting_harvester_refinery_grants_one():
    """RA2 miner logic: no initial harvester; a Refinery grants a free one."""
    state = engine.reset()
    assert int(jnp.sum(state.u_type == 0)) == 0, "must start with NO harvesters"
    # can't train a harvester without a refinery (prereq)
    state = state._replace(credits=state.credits.at[0].set(9000))
    state = engine.step(state, jnp.stack([jnp.array([2, 0, 0], jnp.int32), NOOP]))
    assert int(state.prod_type[0, 2]) == -1, "harvester locked without a refinery"
    # build power -> refinery; refinery grants a free harvester
    state = _build_place(state, 1, 4, 12)
    n0 = int(jnp.sum((state.u_type == 0) & (state.u_owner == 0)))
    state = _build_place(state, 8, 8, 15)          # refinery
    n1 = int(jnp.sum((state.u_type == 0) & (state.u_owner == 0)))
    assert n1 == n0 + 1, f"refinery must grant a free harvester ({n0}->{n1})"


def test_prereq_loss_cancels_queued_units():
    """Destroying a prerequisite building cancels its dependent production."""
    state = engine.reset()
    state = _build_place(state, 1, 4, 12)          # power
    state = _build_place(state, 2, 8, 15)          # barracks
    state = _build_place(state, 4, 8, 11)          # radar (rocketeer needs it)
    state = state._replace(credits=state.credits.at[0].set(9000))
    state = engine.step(state, jnp.stack([jnp.array([2, 2, 0], jnp.int32), NOOP]))  # train rocketeer
    assert int(state.prod_type[0, 0]) == 2, "rocketeer should be in production"
    # destroy the radar (find it, zero its hp)
    ri = int(jnp.argmax((state.b_type == 4) & (state.b_owner == 0)))
    state = state._replace(b_hp=state.b_hp.at[ri].set(0))
    state = engine.step(state, BOTH_NOOP)
    assert int(state.prod_type[0, 0]) == -1, "rocketeer must cancel when radar dies"


def test_unit_gains_veterancy():
    """A unit that deals enough damage ranks up. Target a stationary enemy
    building so the attacker fires continuously (the realistic vet path)."""
    state = engine.reset()
    ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
    ut, uo = ut.at[4].set(3), uo.at[4].set(0)       # p0 tank
    uh = uh.at[4].set(engine.U_HP[3]); ux = ux.at[4].set(10*engine.FP); uy = uy.at[4].set(10*engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    # a big enemy power plant next to the tank (index 4 is free after reset)
    bt = state.b_type.at[4].set(1); bo = state.b_owner.at[4].set(1)
    bh = state.b_hp.at[4].set(99999); bx = state.b_x.at[4].set(int(12.5*engine.FP))
    by = state.b_y.at[4].set(10*engine.FP)
    state = state._replace(b_type=bt, b_owner=bo, b_hp=bh, b_x=bx, b_y=by)
    step = jax.jit(engine.step)
    for _ in range(200):
        state = step(state, BOTH_NOOP)
    assert int(state.u_vet[4]) >= engine.VET1, f"tank should have earned veterancy (vet={int(state.u_vet[4])})"


def test_force_attack_hits_building_behind_screen():
    """cmd 8: force-attack a building even when an enemy unit is closer. Without
    the order the unit auto-acquires the nearer screen unit and the building is
    untouched; with it, the building takes damage."""
    def setup():
        state = engine.reset()
        ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
        ut, uo = ut.at[4].set(1), uo.at[4].set(0)          # p0 rifleman
        uh = uh.at[4].set(engine.U_HP[1])
        ux, uy = ux.at[4].set(10 * engine.FP), uy.at[4].set(10 * engine.FP)
        ut, uo = ut.at[5].set(1), uo.at[5].set(1)          # enemy screen unit (closer)
        uh = uh.at[5].set(engine.U_HP[1])
        ux, uy = ux.at[5].set(11 * engine.FP), uy.at[5].set(10 * engine.FP)
        state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
        bt = state.b_type.at[4].set(1); bo = state.b_owner.at[4].set(1)  # enemy bldg behind
        bh = state.b_hp.at[4].set(99999)
        bx = state.b_x.at[4].set(12 * engine.FP); by = state.b_y.at[4].set(10 * engine.FP)
        return state._replace(b_type=bt, b_owner=bo, b_hp=bh, b_x=bx, b_y=by)

    step = jax.jit(engine.step)
    # baseline: no force order -> auto-acquires the closer screen unit, building safe
    s = setup()
    for _ in range(30):
        s = step(s, BOTH_NOOP)
    assert int(s.b_hp[4]) == 99999, "auto-acquire should have hit the unit, not the building"
    # force-attack the building (encoded target = MAX_U + building index)
    s = setup()
    order = jnp.stack([jnp.array([8, 4, engine.MAX_U + 4], jnp.int32), NOOP])
    s = step(s, order)
    for _ in range(30):
        s = step(s, BOTH_NOOP)
    assert int(s.b_hp[4]) < 99999, "force-attack should have damaged the building behind the screen"


def test_miner_holds_after_commanded_move():
    """A War-Miner given a manual order off ore drives there and HOLDS (mode 4) —
    it does not drift back to auto-mining until commanded again."""
    state = engine.reset()
    ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
    ut, uo = ut.at[4].set(0), uo.at[4].set(0)              # p0 harvester
    uh = uh.at[4].set(engine.U_HP[0])
    ux, uy = ux.at[4].set(20 * engine.FP), uy.at[4].set(20 * engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    order = jnp.stack([jnp.array([5, 4, 24 * 64 + 24], jnp.int32), NOOP])
    state = engine.step(state, order)
    state = _run(state, 120)
    hx, hy = int(state.u_x[4]), int(state.u_y[4])
    assert abs(hx - 24 * engine.FP) < 2 * engine.FP and abs(hy - 24 * engine.FP) < 2 * engine.FP, \
        f"miner should hold near the ordered point, at ({hx/engine.FP:.1f},{hy/engine.FP:.1f})"
    assert int(state.u_mode[4]) == 4, f"miner should be HOLDING (mode 4), got {int(state.u_mode[4])}"


def test_miner_force_crush_then_holds():
    """cmd 8 on a harvester: chase an enemy infantry, crush it, then HOLD."""
    state = engine.reset()
    ut, uo, uh, ux, uy = state.u_type, state.u_owner, state.u_hp, state.u_x, state.u_y
    ut, uo = ut.at[4].set(0), uo.at[4].set(0)              # p0 harvester
    uh = uh.at[4].set(engine.U_HP[0])
    ux, uy = ux.at[4].set(20 * engine.FP), uy.at[4].set(20 * engine.FP)
    ut, uo = ut.at[5].set(1), uo.at[5].set(1)              # enemy rifleman
    uh = uh.at[5].set(engine.U_HP[1])
    ux, uy = ux.at[5].set(24 * engine.FP), uy.at[5].set(20 * engine.FP)
    state = state._replace(u_type=ut, u_owner=uo, u_hp=uh, u_x=ux, u_y=uy)
    order = jnp.stack([jnp.array([8, 4, 5], jnp.int32), NOOP])   # force-attack unit 5
    state = engine.step(state, order)
    state = _run(state, 120)
    assert int(state.u_hp[5]) <= 0 or int(state.u_type[5]) == -1, "miner should have crushed the target"
    assert int(state.u_mode[4]) == 4, f"miner should HOLD after the crush, got mode {int(state.u_mode[4])}"
    assert int(state.u_tgt[4]) == -1, "forced target should clear once the victim dies"


# --------------------------------------------------------------- RL harness
from warpath import rl                                          # noqa: E402


def test_rl_obs_contract():
    """obs is a fixed-width, finite float32 vector for both seats (shared policy)."""
    state = engine.reset()
    for p in (0, 1):
        o = rl.obs(state, p)
        assert o.shape == (rl.OBS_DIM,), f"obs dim {o.shape} != ({rl.OBS_DIM},)"
        assert o.dtype == jnp.float32, f"obs dtype {o.dtype}"
        assert bool(jnp.all(jnp.isfinite(o))), "obs has non-finite entries"


def test_rl_macro_action_all_ids_wellformed():
    """Every macro id maps to a (3,) int32 engine action that steps without
    producing non-finite state (the agent can emit any id safely)."""
    state = engine.reset()._replace(credits=jnp.array([9000, 9000], jnp.int32))
    step = jax.jit(engine.step)
    for m in range(rl.N_ACT):
        a = rl.macro_action(state, 0, jnp.int32(m))
        assert a.shape == (3,) and a.dtype == jnp.int32, f"macro {m} -> {a.shape}/{a.dtype}"
        s2 = step(state, jnp.stack([a, NOOP]))
        assert bool(jnp.all(jnp.isfinite(s2.u_x))) and bool(jnp.all(jnp.isfinite(s2.u_y))), \
            f"macro {m} produced non-finite unit positions"


def test_rl_scripted_selfplay_progresses():
    """A scripted-vs-scripted game runs with finite obs+reward every step and the
    macro layer really drives the engine (the commander develops a base)."""
    state = engine.reset()._replace(credits=jnp.array([4000, 4000], jnp.int32))
    step = jax.jit(engine.step)
    b0_start = int(jnp.sum((state.b_type == 0) & (state.b_owner == 0) & (state.b_hp > 0)))
    for t in range(250):
        prev = state
        a0 = rl.macro_action(state, 0, rl.scripted_macro(state, 0))
        a1 = rl.macro_action(state, 1, rl.scripted_macro(state, 1))
        state = step(state, jnp.stack([a0, a1]))
        assert bool(jnp.isfinite(rl.reward(prev, state, 0))), f"reward non-finite at t={t}"
        assert bool(jnp.all(jnp.isfinite(rl.obs(state, 0)))), f"obs non-finite at t={t}"
        if int(engine.result(state)) >= 0:
            break
    b0_end = int(jnp.sum((state.b_owner == 0) & (state.b_type >= 0) & (state.b_hp > 0)))
    assert b0_end > b0_start, f"scripted commander built nothing ({b0_start}->{b0_end})"


def test_rl_reward_terminal_sign():
    """Destroying the enemy conyard is a strong positive for the winner and a
    strong negative for the loser (terminal term dominates shaping)."""
    state = engine.reset()
    con1 = (state.b_type == 0) & (state.b_owner == 1)
    cur = state._replace(b_hp=jnp.where(con1, 0, state.b_hp))
    assert int(engine.result(cur)) == 0, "p0 should win once p1's conyard is gone"
    assert float(rl.reward(state, cur, 0)) > 0, "winner reward must be positive"
    assert float(rl.reward(state, cur, 1)) < 0, "loser reward must be negative"
