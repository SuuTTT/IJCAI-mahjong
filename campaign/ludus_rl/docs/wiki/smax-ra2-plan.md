# 11 · Games #3 and #4: SMAX micro-battles and the RA2-class RTS

## SMAX (StarCraft-like MARL micro) — days, not weeks

**Foundation**: JaxMARL's SMAX (Apache-2.0) — pure-JAX SMAC recreation, already
installed on our box; our MAPPO baseline hit 1.0 win-rate on 2s3z in 21 GPU
minutes (WO-P1-05, checkpoint committed). Everything vectorizes like Boom.

**Plan (in order):**
1. **Adapter + viewer**: SMAX state → scene schema (plane stage, topdown
   profile) → the SAME three.js renderer. Entities = units (pos/hp/team/type),
   events = shots. Capsules + team rings day one; original unit models via the
   proven generation loop after (marines/zealots analogs need original
   designs — prompt pack when viewer lands).
2. **Replays as records**: (scenario, seed, joint-action log) — the Boom
   replay contract, per-game.
3. **Training slate**: MAPPO across the scenario ladder (2s3z ✓, 3s5z,
   5m_vs_6m, 10m_vs_11m, MMM) — cooperative-vs-heuristic, so the leaderboard
   is per-scenario win-rate with CIs, not head-to-head; upload tier accepts
   SMAX policies once the per-game arch registry lands.
4. **Platform**: game card LIVE on /, spectate page, challenge doc
   (obs/action spec) so agents can be built from the document alone.

## RA2-class RTS — phased, original assets, JAX-first

**Reality check on "open source"**: what's open is the *engine* lineage
(OpenRA and friends recreate C&C-family engines; browser reimplementations
exist). The RA2 *assets and campaigns are proprietary* — same rule as Boom vs
CR: we build mechanics-inspired originals with our own identity. Also, OpenRA
is a C# real-time engine — integrating it means slow, non-vectorized training
outside our JAX stack. Decision: **own JAX engine, RA2-inspired, smallest
competitive slice first** (docs/03 doctrine), sharing Boom's proven core
patterns (int32 state, fixed-point board, replay contract, scene schema).

**"Warpath" P0 slice (the Kaiwu-style 1v1):**
- One-screen map (~48×36 tiles), iso camera profile (renderer ready).
- Economy: ore patches + harvester → refinery credit trickle.
- Production: one construction yard (start), barracks (build), war factory
  (build); 4 unit types: rifleman (cheap), rocketeer (anti-armor), light tank
  (armor), harvester. Power plant as the build-dependency lever.
- Win: destroy the enemy construction yard (or most value at timeout).
- Determinism contract identical to Boom; action space = per-tick macro
  commands (build X, rally, attack-move group to tile) — agent-friendly.
- Assets: the generation loop (iso buildings + vehicles GLBs, portraits).
- Then P1: fog of war, 2 more unit types, naval/air lane, map pool.

**Why this beats adapting OpenRA**: millions of steps/sec for the league
machinery we already have; one renderer; one submission pipeline; no
license ambiguity. OpenRA remains a P3 option for *spectating* human RA2-like
matches if ever needed.

## Sequencing

1. SMAX viewer + adapter (this week) — proves the scene schema on game #3.
2. SMAX training slate on spare GPU capacity (league keeps priority).
3. Warpath engine v1 (economy + production + combat, no fog) + viewer.
4. Warpath PPO baseline + league machinery reuse; asset packs via owner loop.
5. Challenge docs for both; upload tiers behind the per-game arch registry.
