# 03 · Catalog strategy & genre roadmap

## Build 1, adopt many ("env-as-game")
Every adopted env gets: a game page, CI-rated ladder, paired scheduling where the env
allows seeding, replays, calibration bots, and (where sensible) human play.

| Source | Games | Why | License |
|---|---|---|---|
| **Ours** | Boom | flagship + paper | ours |
| **JaxMARL** | SMAX (SC2-micro-like), Overcooked, Hanabi, MPE | MARL researcher credibility; SMAX = the "JAX StarCraft" | Apache-2.0 |
| **PGX** | Go, chess, shogi, backgammon family | classic board ladder, cheap | Apache-2.0 |
| **Craftax** | JAX 2D Minecraft-like (Crafter×NetHack, ~1M steps/s) | the **open-world entry**: seeded duplicate score-race ladder; achievements = quests | MIT |
| **Ours (cheap)** | Texas Hold'em (JAX; tiny state), Chinese Standard Mahjong (port existing engine + official fan library) | huge audiences; poker/mahjong AI scenes | ours |
| **Benchmark-as-game** | gymnax / MuJoCo-Playground tasks as "sport" leaderboards | training IS the game; curriculum quests | Apache-2.0 |

Contribution posture: upstream our paired-duplicate eval wrapper to JaxMARL/Craftax —
free marketing to exactly the right community.

## Open-world: Minecraft path
1. **Now (P1): Craftax** — fits the JAX stack natively; ladder = same world seed for all
   competitors, score/achievement race, optional head-to-head resource contention later.
2. **Later (P4): real 3D Minecraft** (MineRL/MineDojo) as a **scenario showcase category**
   (LLM-agent quests, Voyager-style), not a core vmapped ladder — heavy, non-JAX,
   run-on-demand.

## Genre roadmap for the long wishlist
Rule: every family enters as its **smallest competitive slice**, never full fidelity.

| Family | Inspirations | Entry slice | Phase |
|---|---|---|---|
| RT card battler | CR | Boom | **P0** |
| Cards, imperfect info | YGO-DL, Hold'em, CS Mahjong | Hold'em + Mahjong; YGO-like needs a card-DSL + ~200 original cards (design doc before code) | **P1** (YGO-like P3) |
| RTS micro | SC2 | SMAX adoption | **P1** |
| RTS macro | RA2 | "Boom-RTS": economy + base-building + 2D combat on the Boom engine core | **P3** |
| Hero-commander (RTS×MOBA×open-world) | M&B, Herzog Zwei, Savage, Total War | battle layer 1v1 → conquest campaign (see docs/06) | **P3→P4** |
| MOBA | Dota2, LoL, HoK | 1v1 mid-lite (HoK-Arena-env style); 5v5-lite only after SMAX-scale MARL proven | **P4** |
| Open-world survival | Minecraft | Craftax now; 3D scenarios later | **P1 / P4** |
| Open-world action/RPG | GTA, M&B II (world sim) | NOT ladder-able as such; long-term scenario category | **P4+/out of scope** |

## Adoption checklist (per env)
license ✓ → wrap five-function interface ✓ → determinism suite ✓ → obs schema versioned ✓ →
calibration bots trained ✓ → paired scheduling defined ✓ → replay renderer ✓ → game page.
