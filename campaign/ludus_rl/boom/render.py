"""render_frame(state) -> plain-python dict for generic clients (docs/01 Game SDK).

Engine frame: player 0 owns low y. Clients decide their own screen orientation.
Positions are float tiles (fixed-point / 256)."""

from __future__ import annotations

import numpy as np

from . import engine
from .cards import CARD_NAMES, CARDS
from .engine import FP, TOWER_MAX_HP, TOWER_OWNER, TOWER_X, TOWER_Y


def render_frame(state: engine.State) -> dict:
    s = {f: np.asarray(v) for f, v in zip(state._fields, state)}
    alive = s["u_hp"] > 0
    idx = np.nonzero(alive)[0]
    delay = (s["u_status"][idx] >> 16) & 0xFF
    slow = s["u_status"][idx] & 0xFF
    rage = (s["u_status"][idx] >> 8) & 0xFF
    units = [
        {
            "id": int(i),
            "owner": int(s["u_owner"][i]),
            "card": int(s["u_type"][i]),
            "name": CARD_NAMES[int(s["u_type"][i])],
            "x": float(s["u_x"][i]) / FP,
            "y": float(s["u_y"][i]) / FP,
            "hp": int(s["u_hp"][i]),
            "max_hp": int(CARDS.hp[int(s["u_type"][i])]),
            "air": int(CARDS.air[int(s["u_type"][i])]),
            "deploying": int(d > 0),
            "slow": int(sl > 0),
            "rage": int(r > 0),
        }
        for i, d, sl, r in zip(idx, delay, slow, rage)
    ]
    towers = [
        {
            "owner": int(TOWER_OWNER[t]),
            "kind": "core" if t in (2, 5) else "turret",
            "x": float(TOWER_X[t]) / FP,
            "y": float(TOWER_Y[t]) / FP,
            "hp": int(s["tower_hp"][t]),
            "max_hp": int(TOWER_MAX_HP[t]),
            "active": bool(np.asarray(engine._king_active(state))[t]),
        }
        for t in range(6)
    ]
    return {
        "tick": int(s["tick"]),
        "energy": [float(e) / engine.E_UNIT for e in s["energy"]],
        "units": units,
        "towers": towers,
        "result": int(np.asarray(engine.result(state))),
    }


def _card_info(c: int) -> dict:
    from .cards import ARCHETYPES, CR_REFS

    return {
        "card": c,
        "name": CARD_NAMES[c],
        "cr_ref": CR_REFS[c],
        "cost": int(CARDS.cost[c]),
        "is_spell": int(CARDS.is_spell[c]),
        "archetype": ARCHETYPES[int(CARDS.archetype[c])],
        "hp": int(CARDS.hp[c]),
        "dmg": int(CARDS.dmg[c]),
        "dps": round(int(CARDS.dmg[c]) * 5 / int(CARDS.period[c]), 1)
               if int(CARDS.period[c]) else int(CARDS.dmg[c]),
        "count": int(CARDS.count[c]),
        "air": int(CARDS.air[c]),
        "anywhere": int(CARDS.anywhere[c]),
        "range": round(int(CARDS.range_fp[c]) / 256, 1),
        "speed": int(CARDS.speed[c]),
    }


def hand_info(state: engine.State, player: int) -> dict:
    energy = int(state.energy[player])
    cards = []
    for k, c in enumerate(np.asarray(state.hand[player])):
        d = _card_info(int(c))
        d["slot"] = k
        d["affordable"] = energy >= d["cost"] * engine.E_UNIT
        cards.append(d)
    return {"cards": cards, "next": _card_info(int(state.queue[player, 0])),
            "energy": round(energy / engine.E_UNIT, 2)}


def frame_events(prev: engine.State, new: engine.State,
                 played: list[tuple[int, int, int, int] | None]) -> list[dict]:
    """Visual events derived by diffing two consecutive states (the sim itself has no
    event objects — damage is instant on the attack tick).

    played: per player, (card, x_tile, y_tile, engine-frame) for a card play that was
    applied this tick, else None. Coordinates in the returned events are float tiles,
    engine frame."""
    ev: list[dict] = []
    p_hp, n_hp = np.asarray(prev.u_hp), np.asarray(new.u_hp)
    n_x, n_y = np.asarray(new.u_x) / FP, np.asarray(new.u_y) / FP
    typ = np.asarray(new.u_type)
    owner = np.asarray(new.u_owner)

    # deaths (alive -> dead)
    for i in np.nonzero((p_hp > 0) & (n_hp <= 0))[0]:
        ev.append({"e": "death", "x": float(n_x[i]), "y": float(n_y[i]),
                   "owner": int(owner[i])})

    # floating damage numbers: hp lost this tick, per entity (covers direct hits,
    # splash, spells, death bombs, tower fire — everything)
    for i in np.nonzero((p_hp > 0) & (n_hp < p_hp))[0]:
        ev.append({"e": "dmg", "x": float(n_x[i]), "y": float(n_y[i]),
                   "n": int(p_hp[i] - n_hp[i]), "owner": int(owner[i])})
    p_thp = np.asarray(prev.tower_hp)
    n_thp0 = np.asarray(new.tower_hp)
    tw_x0, tw_y0 = np.asarray(TOWER_X) / FP, np.asarray(TOWER_Y) / FP
    for t in np.nonzero((p_thp > 0) & (n_thp0 < p_thp))[0]:
        ev.append({"e": "dmg", "x": float(tw_x0[t]), "y": float(tw_y0[t]),
                   "n": int(p_thp[t] - n_thp0[t]), "owner": int(TOWER_OWNER[t]),
                   "tower": 1})

    # unit shots: attack cooldown was reset to the card's period this tick
    n_cd = np.asarray(new.u_cd)
    tgt = np.asarray(new.u_tgt)
    period = np.asarray(CARDS.period)[typ]
    fired = (n_hp > 0) & (period > 0) & (n_cd == period) & (tgt >= 0)
    tw_x, tw_y = np.asarray(TOWER_X) / FP, np.asarray(TOWER_Y) / FP
    for i in np.nonzero(fired)[0]:
        t = int(tgt[i])
        tx = tw_x[t - 64] if t >= 64 else n_x[t]
        ty = tw_y[t - 64] if t >= 64 else n_y[t]
        ranged = int(CARDS.range_fp[typ[i]]) > 2 * FP
        ev.append({"e": "shot" if ranged else "melee", "owner": int(owner[i]),
                   "x0": float(n_x[i]), "y0": float(n_y[i]),
                   "x1": float(tx), "y1": float(ty)})

    # tower shots: tower cooldown reset; victim is the tower's sticky target
    n_tcd = np.asarray(new.tower_cd)
    n_thp = np.asarray(new.tower_hp)
    t_tgt = np.asarray(new.tower_tgt)
    for t in np.nonzero((n_tcd == engine.TOWER_PERIOD_ARR) & (n_thp > 0)
                        & (t_tgt >= 0))[0]:
        v = int(t_tgt[t])
        ev.append({"e": "shot", "tower": 1, "owner": int(TOWER_OWNER[t]),
                   "x0": float(tw_x[t]), "y0": float(tw_y[t]),
                   "x1": float(n_x[v]), "y1": float(n_y[v])})

    # tower falls
    for t in np.nonzero((np.asarray(prev.tower_hp) > 0) & (n_thp <= 0))[0]:
        ev.append({"e": "tower_fall", "x": float(tw_x[t]), "y": float(tw_y[t]),
                   "owner": int(TOWER_OWNER[t])})

    # card plays: deploys thump; instant spells blast now; lobbed spells CAST
    # (a projectile flies from the caster's king tower and impacts on arrival)
    for p, act in enumerate(played):
        if act is None:
            continue
        card, ax, ay = act
        if int(CARDS.is_spell[card]):
            delay = int(CARDS.spell_delay[card])
            if delay == 0:
                ev.append({"e": "spell", "owner": p, "x": ax + 0.5, "y": ay + 0.5,
                           "r": float(CARDS.splash_fp[card]) / FP,
                           "effect": int(CARDS.effect[card])})
            else:
                kx, ky = (9.0, 3.0) if p == 0 else (9.0, 29.0)
                ev.append({"e": "cast", "owner": p, "card": card,
                           "x0": kx, "y0": ky, "x": ax + 0.5, "y": ay + 0.5,
                           "eta": delay})
        else:
            ev.append({"e": "deploy", "owner": p, "x": ax + 0.5, "y": ay + 0.5})

    # lobbed-spell impacts: pending rows that hit zero this tick
    p_sp = np.asarray(prev.spells)
    for row in p_sp:
        card, owner_, cx, cy, tl = (int(v) for v in row)
        if card >= 0 and tl <= 1:
            ev.append({"e": "spell", "owner": owner_, "x": cx / FP, "y": cy / FP,
                       "r": float(CARDS.splash_fp[card]) / FP,
                       "effect": int(CARDS.effect[card])})
    return ev
