"""Card table loader: cards.csv -> immutable int32 arrays for the JAX engine.

Balance source (v3): every card carries the tournament-standard (level 11) numeric
statistics of its closest Clash Royale analog, extracted from the RoyaleAPI
community dataset (cr-api-data) by benchmarks/gen_cards_v3.py. Game statistics are
unprotectable functional facts; all NAMES and ART here are original — `cr_ref`
records the analog purely as factual provenance. Mechanics our engine cannot
express (charge, shields, death-spawns, spawners, knockback, chain, pierce) are
documented per-card in the generator; such cards behave per our engine.
Refinements/balance patches are data-only commits to this CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np

FP = 256  # fixed-point units per tile

ARCHETYPES = (
    "tank", "swarm", "splash", "ranged", "air", "antiair",
    "building", "spell_dmg", "spell_util", "wincon", "cycle", "support",
)

# effect codes (spells + on-hit); STUN halts victims and resets their target locks
EFFECT_NONE, EFFECT_SLOW, EFFECT_RAGE, EFFECT_STUN, EFFECT_PULL = 0, 1, 2, 3, 4
# aura codes (legacy; v3 set has no aura units)
AURA_NONE, AURA_RAGE, AURA_HEAL = 0, 1, 2


class CardTable(NamedTuple):
    """Column-wise card data. All int32 (fixed-point where noted)."""

    cost: np.ndarray          # energy cost, 1..9
    hp: np.ndarray            # max hp (0 for spells)
    dmg: np.ndarray           # damage per hit
    period: np.ndarray        # ticks between hits (0 = never attacks)
    speed: np.ndarray         # fp per tick (0 = stationary)
    range_fp: np.ndarray      # attack range, fp
    count: np.ndarray         # bodies spawned per play (<= 15)
    air: np.ndarray           # 1 = flying unit
    targets_air: np.ndarray   # 1 = can hit flying units
    bldg_only: np.ndarray     # 1 = only targets towers/buildings
    splash_fp: np.ndarray     # splash radius, fp (0 = single target)
    is_spell: np.ndarray      # 1 = instant spell, spawns nothing
    effect: np.ndarray        # spell status effect code
    duration: np.ndarray      # spell status duration, ticks
    decay: np.ndarray         # building self-damage per tick (lifetime)
    aura_type: np.ndarray     # legacy aura code (all 0 in v3)
    aura_power: np.ndarray
    aura_radius_fp: np.ndarray
    hit_effect: np.ndarray    # status applied to victims ON HIT (slow/stun)
    hit_dur: np.ndarray       # on-hit status duration, ticks
    suicide: np.ndarray       # 1 = unit dies after its first attack (spirits)
    death_dmg: np.ndarray     # splash damage dealt on death (bombs)
    death_r_fp: np.ndarray    # death splash radius, fp
    anywhere: np.ndarray      # 1 = deployable on any tile (miner-like)
    tower_pct: np.ndarray     # % of dmg dealt to towers by spells (100 for units)
    kb_fp: np.ndarray         # spell knockback distance, fp (fireball/snowball family)
    col_r_fp: np.ndarray      # body collision radius, fp
    mass: np.ndarray          # collision mass (heavier pushes lighter; buildings ~inf)
    jumps: np.ndarray         # 1 = crosses the river without bridges (hog family)
    spell_delay: np.ndarray   # spell flight time, ticks (0/1 = effectively instant)
    no_tower: np.ndarray      # 1 = never targets towers (CR defensive buildings)
    charge: np.ndarray        # 1 = prince family: run-up doubles the first hit
    hook: np.ndarray          # 1 = fisherman family: each hit drags the victim adjacent
    spawn_type: np.ndarray    # spell that spawns units: unit id to spawn (-1 = none)
    archetype: np.ndarray     # index into ARCHETYPES (for obs planes)


def load_cards(path: str | Path | None = None) -> tuple[CardTable, list[str], list[str]]:
    """Load the card table. Returns (columns, names, cr_refs). Loud-fails on bad data.

    TEST-ONLY: BOOM_CARDS_PATCH="id:column:value[;...]" mutates stats after load —
    used by the ladder's synthetic-bug calibration drill (docs/04 §1). Never set in
    production; when set, a loud banner prints and the patch joins the judge hash."""
    path = Path(path) if path else Path(__file__).parent / "cards.csv"
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 66, f"expected 66 cards, got {len(rows)}"  # v16: +Timberoll +Marrowkeg +Brawlkeg

    import os
    patch = os.environ.get("BOOM_CARDS_PATCH", "")
    if patch:
        print(f"!!! BOOM_CARDS_PATCH ACTIVE (test-only): {patch} !!!", flush=True)
        for spec in patch.split(";"):
            cid, col, val = spec.split(":")
            rows[int(cid)][col] = val

    def col(key: str, scale: float = 1.0) -> np.ndarray:
        return np.array([int(round(float(r[key]) * scale)) for r in rows], dtype=np.int32)

    ids = col("id")
    assert (ids == np.arange(len(rows))).all(), "card ids must be 0..N in order"
    names = [r["name"] for r in rows]
    cr_refs = [r["cr_ref"] for r in rows]
    arch = np.array([ARCHETYPES.index(r["archetype"]) for r in rows], dtype=np.int32)

    t = CardTable(
        cost=col("cost"), hp=col("hp"), dmg=col("dmg"), period=col("period"),
        speed=col("speed"), range_fp=col("range", FP), count=col("count"),
        air=col("air"), targets_air=col("targets_air"), bldg_only=col("bldg_only"),
        splash_fp=col("splash", FP), is_spell=col("is_spell"), effect=col("effect"),
        duration=col("duration"), decay=col("decay"), aura_type=col("aura_type"),
        aura_power=col("aura_power"), aura_radius_fp=col("aura_radius", FP),
        hit_effect=col("hit_effect"), hit_dur=col("hit_dur"), suicide=col("suicide"),
        death_dmg=col("death_dmg"), death_r_fp=col("death_r", FP),
        anywhere=col("anywhere"), tower_pct=col("tower_pct"),
        kb_fp=col("kb", FP),
        col_r_fp=col("col_r", FP), mass=col("mass"), jumps=col("jumps"),
        spell_delay=col("spell_delay"), no_tower=col("no_tower"),
        charge=col("charge"),
        hook=col("hook"),
        spawn_type=col("spawn_type"),
        archetype=arch,
    )
    _validate(t)
    return t, names, cr_refs


def _validate(t: CardTable) -> None:
    assert ((t.cost >= 1) & (t.cost <= 9)).all()
    units = t.is_spell == 0
    assert (t.hp[units] > 0).all(), "units need hp"
    assert (t.hp[~units] == 0).all(), "spells have no hp"
    attackers = units & (t.dmg > 0)
    assert (t.period[attackers] > 0).all(), "attacking units need a period"
    assert ((t.count >= 1) & (t.count <= 15)).all()
    assert ((t.tower_pct >= 20) & (t.tower_pct <= 100)).all()
    assert (t.hit_dur[t.hit_effect > 0] > 0).all()
    # archetype coverage: every archetype has >= 2 cards (docs/02 contract)
    counts = np.bincount(t.archetype, minlength=len(ARCHETYPES))
    assert (counts >= 2).all(), f"archetype coverage broken: {counts}"


CARDS, CARD_NAMES, CR_REFS = load_cards()

# Default decks (classic shapes): A = cheap cycle around Ramhound; B = big-tank beatdown
DECK_A = np.array([39, 37, 6, 47, 25, 52, 56, 3], dtype=np.int32)
DECK_B = np.array([1, 20, 14, 12, 53, 55, 31, 49], dtype=np.int32)
