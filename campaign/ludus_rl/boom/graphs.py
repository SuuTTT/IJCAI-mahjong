"""Composed rule-policies (Composer Tier-3 v0): shared interpreter.

A graph is an ordered rule list; each tick the first rule whose condition
holds AND whose card is affordable fires. Used by the play server (live bots)
and the ladder daemon (rated matches) so composed bots behave identically
everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import boom.cards as cards_mod
from boom import engine

GRAPHS_DIR = Path("/root/ludus_graphs")

SENSORS = ("elixir", "enemy_in_my_half", "my_tower_min_pct",
           "enemy_tower_min_pct", "seconds")

# player-frame tile presets: always in one's own half -> always placeable
SPOTS = {"left_bridge": (4, 14), "right_bridge": (13, 14),
         "center_back": (9, 5), "left_back": (3, 8), "right_back": (14, 8),
         "king_front": (9, 10)}

NOOP = [4, 0, 0]
BOT_PERIOD = 5


def load_rules(name: str, graphs_dir: Path | None = None) -> list[dict]:
    d = graphs_dir or GRAPHS_DIR
    return json.loads((d / f"{name}.json").read_text())["rules"]


def sensors_of(state, seat: int):
    s = {f: np.asarray(v) for f, v in zip(state._fields, state)}
    opp = 1 - seat
    alive = s["u_hp"] > 0
    enemy = alive & (s["u_owner"] == opp)
    my_half = (s["u_y"] // 256 <= 15) if seat == 0 else (s["u_y"] // 256 >= 16)
    towers_me = [i for i in range(6) if engine.TOWER_OWNER[i] == seat]
    towers_opp = [i for i in range(6) if engine.TOWER_OWNER[i] == opp]
    hp = s["tower_hp"]
    maxhp = np.asarray(engine.TOWER_MAX_HP)
    hand = s["hand"][seat]
    costs = np.asarray(cards_mod.CARDS.cost)[hand]
    return {
        "elixir": float(s["energy"][seat]) / engine.E_UNIT,
        "enemy_in_my_half": int((enemy & my_half).sum()),
        "my_tower_min_pct": float((hp[towers_me] / maxhp[towers_me]).min()),
        "enemy_tower_min_pct": float((hp[towers_opp] / maxhp[towers_opp]).min()),
        "seconds": float(s["tick"]) / 5.0,
    }, costs


def make_policy(rules: list[dict], seat: int):
    """-> act(key, state, tick) -> [slot, x, y] in the player frame."""
    def act(key, state, tick):
        if tick % BOT_PERIOD != 0:
            return NOOP
        sensors, costs = sensors_of(state, seat)
        elixir = sensors["elixir"]
        for r in rules:
            v = sensors.get(r["sensor"], 0.0)
            ok = v > r["value"] if r["op"] == ">" else v < r["value"]
            if not ok:
                continue
            slot = int(np.argmin(costs)) if r["slot"] == "cheapest" else int(r["slot"])
            if costs[slot] > elixir:
                continue
            x, y = SPOTS.get(r["at"], (9, 10))
            return [slot, x, y]
        return NOOP
    return act
