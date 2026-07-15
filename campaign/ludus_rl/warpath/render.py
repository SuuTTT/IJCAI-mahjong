"""Warpath frame extraction for viewers (schema-shaped, docs/09).

PURE NUMPY on purpose: a frame is built twice per live tick, and any jax call
here would sync against whatever is hogging the accelerator (the league keeps
the GPU at 100%). Everything derives from the already-transferred state dict.
"""

from __future__ import annotations

import numpy as np

from . import engine

_U_HP = np.asarray(engine.U_HP)
_U_PERIOD = np.asarray(engine.U_PERIOD)
_U_RANGE_FP = np.asarray(engine.U_RANGE_FP)
_U_BUILD_T = np.asarray(engine.U_BUILD_T)
_U_PREREQ = np.asarray(engine.U_PREREQ)
_B_HP = np.asarray(engine.B_HP)
_B_PERIOD = np.asarray(engine.B_PERIOD)
_B_RANGE_FP = np.asarray(engine.B_RANGE_FP)
_B_PREREQ = np.asarray(engine.B_PREREQ)
_B_POWER = np.asarray(engine.B_POWER)
_B_BUILD_T = np.asarray(engine.B_BUILD_T)
_ORE_X = np.asarray(engine.ORE_X)
_ORE_Y = np.asarray(engine.ORE_Y)


def _np_owned_mask(s, p):
    own = (s["b_type"] >= 0) & (s["b_owner"] == p) & (s["b_hp"] > 0) & (s["b_prog"] == 0)
    mask = 0
    for t in np.unique(s["b_type"][own]):
        mask |= 1 << int(t)
    return mask


def _np_power(s, p):
    own = (s["b_type"] >= 0) & (s["b_owner"] == p) & (s["b_hp"] > 0) & (s["b_prog"] == 0)
    return int(np.sum(np.where(own, _B_POWER[np.maximum(s["b_type"], 0)], 0)))


def frame(state) -> dict:
    s = {k: np.asarray(v) for k, v in state._asdict().items()}
    units = []
    alive_u = (s["u_type"] >= 0) & (s["u_hp"] > 0)
    alive_b = (s["b_type"] >= 0) & (s["b_hp"] > 0)
    for i in range(engine.MAX_U):
        if not alive_u[i]:
            continue
        t = int(s["u_type"][i])
        fired = bool(s["u_cd"][i] == int(_U_PERIOD[t]) and t >= 1)
        u = {
            "id": i, "type": t, "owner": int(s["u_owner"][i]),
            "x": float(s["u_x"][i]) / engine.FP, "y": float(s["u_y"][i]) / engine.FP,
            "hp": int(s["u_hp"][i]), "max": int(_U_HP[t]),
            "mode": int(s["u_mode"][i]),
            "air": int(np.asarray(engine.U_AIR)[t]),
            "vet": (1 if s["u_vet"][i] >= engine.VET2 else 0) + (1 if s["u_vet"][i] >= engine.VET1 else 0),
            "fired": fired,
        }
        if fired:
            # tracer target = nearest enemy (matches the engine's argmin choice).
            # Only draw the beam if that target is within WEAPON range: when a shot
            # kills its target on the same tick, the real victim is already gone and
            # the nearest survivor is the far enemy base — drawing to it produced a
            # phantom cross-map tracer. Beyond range -> muzzle flash only.
            foe_u = alive_u & (s["u_owner"] != s["u_owner"][i])
            du = np.where(foe_u, (s["u_x"] - s["u_x"][i]) ** 2
                          + (s["u_y"] - s["u_y"][i]) ** 2, 2 ** 30)
            foe_b = alive_b & (s["b_owner"] != s["u_owner"][i])
            db = np.where(foe_b, (s["b_x"] - s["u_x"][i]) ** 2
                          + (s["b_y"] - s["u_y"][i]) ** 2, 2 ** 30)
            best_d2 = min(int(du.min()), int(db.min()))
            rng2 = int(_U_RANGE_FP[t]) ** 2
            if best_d2 <= rng2:
                if du.min() <= db.min():
                    j = int(np.argmin(du))
                    u["tx"], u["ty"] = float(s["u_x"][j]) / engine.FP, float(s["u_y"][j]) / engine.FP
                else:
                    j = int(np.argmin(db))
                    u["tx"], u["ty"] = float(s["b_x"][j]) / engine.FP, float(s["b_y"][j]) / engine.FP
        units.append(u)
    bldgs = []
    constructing = None
    for i in range(engine.MAX_B):
        if s["b_type"][i] < 0 or s["b_hp"][i] <= 0:
            continue
        t = int(s["b_type"][i])
        total = max(int(_B_BUILD_T[t]), 1)
        prog = 100 - int(100 * s["b_prog"][i] / total) if s["b_prog"][i] > 0 else 100
        if s["b_prog"][i] > 0 and int(s["b_owner"][i]) == 0:
            constructing = {"type": t, "pct": prog,
                            "secs": int(np.ceil(s["b_prog"][i] / engine.TICKS_PER_S))}
        bl = {"id": i, "type": t, "owner": int(s["b_owner"][i]),
              "x": float(s["b_x"][i]) / engine.FP, "y": float(s["b_y"][i]) / engine.FP,
              "hp": int(s["b_hp"][i]), "max": int(_B_HP[t]), "prog": prog}
        if t >= 6 and int(s["b_cd"][i]) == int(_B_PERIOD[t]):
            au = (s["u_type"] >= 0) & (s["u_hp"] > 0) & (s["u_owner"] != s["b_owner"][i])
            du = np.where(au, (s["u_x"] - s["b_x"][i]) ** 2 + (s["u_y"] - s["b_y"][i]) ** 2, 2 ** 30)
            if du.min() <= int(_B_RANGE_FP[t]) ** 2:
                j = int(np.argmin(du))
                bl["tx"], bl["ty"] = float(s["u_x"][j]) / engine.FP, float(s["u_y"][j]) / engine.FP
        bldgs.append(bl)
    ore = [{"x": float(x) / engine.FP, "y": float(y) / engine.FP, "amt": int(a)}
           for x, y, a in zip(_ORE_X, _ORE_Y, s["ore"])]
    om = _np_owned_mask(s, 0)
    powers = [_np_power(s, 0), _np_power(s, 1)]
    # RA2 sidebar construction: pending building for p0
    pend_t = int(s["pend_t"][0]); pend_ty = int(s["pend_type"][0])
    pending = None
    if pend_ty >= 0:
        total = max(int(np.asarray(engine.B_BUILD_T)[pend_ty]), 1)
        pending = {"type": pend_ty, "ready": pend_t == 0,
                   "pct": 100 - int(100 * pend_t / total),
                   "secs": int(np.ceil(pend_t / engine.TICKS_PER_S))}
    busy_building = pend_ty >= 0
    b_avail = []
    for bt in range(9):
        pre = int(_B_PREREQ[bt])
        need = [t for t in range(9) if (pre >> t) & 1 and not (om >> t) & 1]
        b_avail.append({"ok": (om & pre) == pre and bt != 0 and not busy_building,
                        "need": need})
    u_avail = []
    SRC = [0, 2, 2, 3, 3, 3]
    for ut in range(6):
        pre = int(_U_PREREQ[ut])
        has_src = bool((om >> SRC[ut]) & 1)
        need = [t for t in range(9) if (pre >> t) & 1 and not (om >> t) & 1]
        if not has_src:
            need = [SRC[ut]] + need
        u_avail.append({"ok": has_src and (om & pre) == pre, "need": need})
    prod_pct = [[
        0 if s["prod_type"][p][q] < 0 else int(100 * (1 - s["prod_t"][p][q] /
            max(1, int(_U_BUILD_T[s["prod_type"][p][q]]) * (1 if powers[p] >= 0 else 2))))
        for q in range(3)] for p in range(2)]
    return {
        "type": "frame", "tick": int(s["tick"]),
        "power": powers,
        "avail": {"build": b_avail, "train": u_avail},
        "constructing": constructing,
        "pending": pending,
        "credits": [int(c) for c in s["credits"]],
        "prod": [[int(t) for t in row] for row in s["prod_type"]],
        "prod_t": [[int(t) for t in row] for row in s["prod_t"]],
        "prod_pct": prod_pct,
        "queue": [[int(np.sum(s["prod_q"][p][q] >= 0)) for q in range(3)]
                  for p in range(2)],
        "units": units, "buildings": bldgs, "ore": ore,
        "result": int(np.asarray(engine.result(state))),
    }
