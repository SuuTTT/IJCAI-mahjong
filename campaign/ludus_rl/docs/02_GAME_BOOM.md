# 02 · Boom — flagship game design (v1)

An original real-time card-battler. Mechanics inspired by the lane-battler genre;
**all names and art are original.** As of v2, numeric balance (costs, hp, damage,
hit speeds, movement speeds, ranges, tower stats, targeting rules) references Clash
Royale tournament-standard values — game statistics are unprotectable facts; no
names, assets, or art are copied.

## Match shape
- 1v1, 3 minutes + 60 s overtime (sudden death: first tower). 2D board 18×32 tiles,
  2 lanes, a river with 2 bridges.
- Each player: 1 **Core** (main tower) + 2 **Turrets**. Destroy more turrets / the Core.
- **Energy**: 0–10, +1 per 2.8 s (double in last 60 s). Cards cost 1–9.
- **Deck**: 8 cards, hand of 4, fixed cycle (played card goes to back of queue).
- Tick rate: 5 ticks/s (200 ms). Actions per tick: `(card_slot, x, y)` or no-op.
  Placement legal only on own half (spells anywhere).

## State (int32/fixed-point, JAX)
```
units:   [MAX_UNITS=64] × (owner, type, hp, x_fp, y_fp, target_id, cooldown, status)
towers:  6 × (owner, kind, hp)
economy: 2 × (energy_fp, hand[4], queue[4], cooldowns)
clock:   tick, overtime_flag, prng_key
```
Movement/pathing v1: lane-follow + local steering (no full A*): deterministic integer math.
Targeting: nearest-in-range with fixed tie-break (id order) — deterministic.

## Card set v1 — 60 originals covering the archetype space
Design to the **archetype coverage matrix** (each cell ≥2 cards):

| Archetype | Role | Examples (original names) |
|---|---|---|
| Tank | soak, slow, high HP | Bulwark, Golemite |
| Swarm | many cheap bodies | Ratpack, Sporelings |
| Splash | AoE ground | Emberwitch, Mortar Crab |
| Ranged DPS | single-target | Longshot, Hexarcher |
| Air | flies over ground | Zephyrling, Skyray |
| Anti-air | targets air | Spikethrower, Flakbot |
| Building | defensive structure | Watchpost, Tesla Bloom |
| Spell-damage | direct dmg AoE | Fireburst, Shockwave |
| Spell-utility | slow/rage/clone | Frostfield, Overclock |
| Win-condition | tower-focused | Ramhound, Siege Snail |
| Cheap cycle | 1–2 cost filler | Pebbling, Dart Frog |
| Support/aura | buffs nearby | Bannerbeast, Chorus Wisp |

Stat design rule: every card gets (cost, hp, dps, speed, range, count, air_flag,
targets_air, splash_r) in a single `cards.csv` — balance patches are data-only commits.

## Observation (v1, versioned `obs/v1`)
- Spatial: 18×32×C planes (unit type one-hot ×2 owners, hp buckets, towers) — C≈24.
- Vector: energy, hand one-hots, cycle position, clock, tower HPs.
- Full state minus opponent hand/queue (imperfect info = hidden cycle).

## Research hooks (day one)
- **Duplicate mode:** mirrored decks + mirrored energy/spawn RNG across the pair of games
  (A-side/B-side); placement decided on aggregate — the variance-reduction trick that makes
  ±1% effects measurable. Novel for RTS-likes.
- Per-tick obs tensors exportable for BC; balance patches as natural experiments.
- Built-in calibration bots at fixed ratings: `rule_v0` (scripted heuristic),
  `bc_v0` (BC on playtests), `ppo_v0` (self-play PPO).

## Explicit non-goals v1
No deck-building ladder (fixed meta decks first), no 3+ lanes, no champions/heroes,
no progression/unlocks (research fairness), no monetized cards ever (data/compute is the
business, not pay-to-win).
