"""Boom: deterministic JAX real-time card-battler (Ludus flagship game)."""

from .cards import CARD_NAMES, CARDS, DECK_A, DECK_B
from .engine import (
    H,
    MAX_UNITS,
    OBS_C,
    OBS_VEC,
    RESULT_DRAW,
    RESULT_ONGOING,
    RESULT_P0,
    RESULT_P1,
    TICKS_MAX,
    TICKS_REG,
    W,
    Obs,
    State,
    legal,
    observe,
    reset,
    result,
    step,
)

ENV_VERSION = "boom/v16"   # v8: spell flight times + tower freeze/stun +
# short spirit leap. v7: exact CR tower geometry (6.5/25.5) + crossed-side
# engagement, spirit leap, quarter-overlap soft collision. v6: impassable river,
# global building attraction. v5: collision physics + river-jumpers.
# v4: knockback, pockets, 3x OT elixir, tiebreak. v3: exact tournament stats.

__all__ = [
    "CARDS", "CARD_NAMES", "DECK_A", "DECK_B", "ENV_VERSION",
    "H", "W", "MAX_UNITS", "OBS_C", "OBS_VEC",
    "RESULT_DRAW", "RESULT_ONGOING", "RESULT_P0", "RESULT_P1",
    "TICKS_MAX", "TICKS_REG",
    "Obs", "State", "legal", "observe", "reset", "result", "step",
]
