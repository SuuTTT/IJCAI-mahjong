# Boom ↔ Clash Royale mechanics audit

The living checklist: every documented CR mechanic, whether Boom implements it,
and the test that pins it. Sources: community wiki texts, RoyaleAPI mechanics
docs, and playtester reports (stats are unprotectable facts; names/art here are
original). Status: ✅ implemented+tested · 🟡 partial/approximated · ❌ not yet.

## Board & towers

| mechanic | CR rule | status | pinned by |
|---|---|---|---|
| Arena size | 18×32 tiles, river splits at mid, two 2-tile bridges | ✅ | test_river_impassable, map constants |
| Princess tower range | 7.5 tiles | ✅ v9 | test_princess_tower_geometry |
| King tower range | 7.0 tiles | ✅ v9 | test_king_range_is_seven_tiles |
| Tower engagement side | towers only engage units that crossed onto their side; never over river/bridge | ✅ | test_frostsprite_lands_freeze, crossing tests |
| King activation | king fires only after taking damage or a princess falls | ✅ | test_king_activation |
| Tower HP/damage/hit speed | princess 3052/109/0.8s, king 4824/109/1.0s (level 11) | ✅ | cards.csv constants, sanity artifact |
| First-hit delay | towers take a short aim delay on new targets | ✅ v10 (~0.4s on fresh targets) | test_tower_first_hit_aim_delay |
| Pocket deploy | after a princess falls, deploy unlocks rows up to ~4.5 tiles before the king | ✅ | test_pocket_depth_capped |
| King march bias + lane pathing | units path via bridges (true-path distance), princess preferred over king | ✅ v11 (bridge-aware linear metric for march; acquisition still euclidean) | test_pocket_hog_prefers_princess_over_king |

## Units

| mechanic | CR rule | status | pinned by |
|---|---|---|---|
| Sticky targeting | units keep their lock until target dies or a stun breaks it | ✅ | test_sticky_lock |
| Zap/stun retarget | stuns reset locks | ✅ | test_stun_resets_lock |
| Building-targeters | ignore troops; continuously seek nearest structure (new buildings pull them) | ✅ | test_new_building_pulls_marching_giant |
| Defensive buildings vs towers | player buildings never attack towers | ✅ | test_defensive_building_never_attacks_towers |
| Deploy time | 1s before a unit acts | ✅ | deploy-delay status bits |
| Collision & mass | units push smaller units; heavier units displace lighter | 🟡 (mass-weighted ¼-overlap resolution) | test_collision_separation |
| Knockback | fireball/snowball/etc push units | ✅ | test_knockback |
| Building knockback immunity | buildings are anchored — never pushed | ✅ v10 | test_building_immune_to_knockback |
| Hog river jump | hog-class jumps the river | ✅ | test_hog_jump |
| Spirit leap | spirits leap a short distance to connect | ✅ v8 | test_frostsprite_lands_freeze |
| Air/ground layers | air ignores collisions, only anti-air hits it | ✅ | targets_air/air flags + tests |
| Aggro/sight range | ~5.5 tile sight, building-targeters see structures globally | 🟡 (sight ≥ max(5.5, reach+2)) | acquisition tests |
| Retarget on kill | pick nearest eligible after target dies | ✅ | acquisition logic |
| Charge mechanics (prince/ram) | double damage after charge distance | ✅ v10 (new card: Lancer; 3.5-tile run-up, +60% speed, 2× first hit) | test_charge_doubles_first_hit |
| Death spawns (e.g. molten golem) | spawn units on death | ✅ | death_dmg/death_r + spawn cards |

## Spells

| mechanic | CR rule | status | pinned by |
|---|---|---|---|
| Flight time | fireball ~1s, arrows ~1s zoned, rocket slower | ✅ v8 | test_spell_flight_timing |
| Tower damage reduction | spells deal ~35% reduced damage to towers | ✅ | tower_pct column |
| Freeze stops towers | freeze/stun affects towers too | ✅ v8 | test_freeze_stuns_tower |
| Predictive aim | none — spells land where cast | ✅ | (by construction) |

## Economy & match flow

| mechanic | CR rule | status | pinned by |
|---|---|---|---|
| Elixir | 1 per 2.8s; 2× after 2:00; 3× in overtime; cap 10; start 5 | ✅ v10 | test_elixir_starts_at_five |
| Match clock | 3:00 regulation + up to 2:00 overtime | ✅ | TICKS_REG/TICKS_MAX |
| Overtime entry | tied towers at 3:00 → first-tower-wins sudden death | ✅ | test_regulation_tower_lead_wins |
| Tiebreak | tower count, then lowest-tower-HP comparison | ✅ | result() tests |
| Card cycle | 8-card deck, 4-card hand, played card goes to back | ✅ | test_card_cycle |
| Same-card queue | next copy can't be drawn immediately | ✅ | cycle logic |

## Known open items (next fixes)

1. Acquisition-path distance (march is bridge-aware since v11; target
   acquisition still compares euclidean).
2. Splash area shape nuances (radii are per-card from the cr-api-data source;
   projectile-vs-area shape differences are approximated).

*Report a mismatch: play it, note the replay id from /replays, and describe the
expected CR behavior — each confirmed report becomes a row and a test here.*

| Walk-retarget (kiting: interposed body pulls a walking troop) | ✅ v14 | test_walking_unit_retargets_to_interposed_body |

## Targeting & pathfinding rules (CR reference, owner-supplied)

The mental model: **filter valid targets → choose a target in sight → path
toward it → retarget only when needed.**

| Rule | Status | Pin |
|---|---|---|
| Target filters: ground/air/both per card | ✅ | targets_air column |
| Building-targeters ignore troops; buildings pull them globally | ✅ | test_pocket_hog_prefers_princess_over_king |
| Lock holds only while ENGAGED; walking troops re-evaluate (kiting, ice-golem pulls, tank/support splitting) | ✅ v14 | test_walking_unit_retargets_to_interposed_body |
| Stun/freeze forces retarget on resume | ✅ | stun handler resets locks |
| Bodies occupy space; units path around collision (incl. tower footprints) | ✅ v12 | test_unit_slides_around_own_tower |
| Knockback: buildings anchored; heavies (mass ≥ 10) immune; water rule holds | ✅ v13 | test_heavy_unit_immune_to_knockback, test_knockback_cannot_push_unit_into_river |
| Per-card SIGHT ranges (hog 9.5, balloon 7.7 vs standard 5.5) | ⚠ TODO | uniform 5.5 + range floor today |
| Ground pathing via bridges; fliers cross anywhere | ✅ | _path_lin16 bridge distance |

## King Tower activation (CR reference, owner-supplied)

**Rule: the king starts asleep and wakes when it takes ANY damage (troop hit,
spell chip, splash, pierce, pull-induced) or when an own princess tower
falls.** Ranged win conditions that never enter king range are unaffected;
Goblin-Barrel/Graveyard-class cards get much weaker after activation.

| Rule | Status | Pin |
|---|---|---|
| King asleep until damaged or own princess falls | ✅ | _king_active (hp-based: covers ALL damage sources) |
| Spell chip activates king | ✅ v15 pin | test_spell_damage_activates_king |
| Activation visible in 3D (king dims asleep / glows awake) | ✅ v15 | render towers carry `active` |
| **Whirlgale** (Tornado analog, id 61): 3-elixir pull-to-center spell — drags everything incl. heavies, buildings anchored, 30% tower damage | ✅ v15 | test_whirlgale_pulls_heavy_but_not_building |
| **Harpooner** (Fisherman analog, id 62): 3-elixir hook unit — each hit drags the victim adjacent; cross-river hooks land on the bank | ✅ v15 | test_harpooner_drags_victim_adjacent |
| Activation plays: whirl a lone Ramhound into the king; Harpooner in front of the king hooks bridge units inward | 🎮 try it | |
