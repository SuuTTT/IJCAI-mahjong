# 06 · Hero-Commander — the RTS × MOBA × open-world hybrid (design, P3→P4)

Working title: **Warbound** (original IP). The answer to "combine RTS, MOBA, and
open-world like Mount & Blade": a **hero-commander** game — you ARE one hero on the field
(MOBA micro) AND you command squads (RTS macro), with a persistent conquest campaign as
the open-world loop.

Genre precedents (mechanics-inspiration only): Mount & Blade (hero + troops + campaign),
Herzog Zwei (act + command, the MOBA ancestor), Savage / Natural Selection (player+commander
hybrid), Total War (hero-general + army). None exist as a clean, fast RL environment —
that is the opportunity.

## Why this is the best research object on the roadmap
- **Natural hierarchy benchmark:** two action timescales in one game — hero actions
  (per-tick micro) vs command actions (squad orders, sparse). Perfect testbed for
  hierarchical RL / options — connects directly to our hierarchical-abstraction research
  line (sample-efficiency of command-level abstraction is a paper by itself).
- **Asymmetric human-agent teams:** human hero + AI squads, or AI hero + human commander —
  a data/HRI goldmine no platform has.
- **Smooth genre bridge:** reuses the Boom engine (units, pathing, combat); adds economy
  from Boom-RTS; the campaign layer adds open-world persistence without 3D.

## Phase 1 (P3): Battle layer — 1v1, JAX, ~5 min
- Board ~48×48 tiles, open field + 2–3 capture points, deterministic tick sim @5/s.
- Each side: 1 **Hero** (choose 1 of 6 classes: abilities on cooldown, XP/levels within the
  battle — the MOBA layer) + up to 6 **squads** (recruited pre-battle from a points budget:
  spear/shield/archer/cavalry/siege/skirmisher — the RTS layer).
- Command verbs (sparse): move-to, attack-move, hold, follow-hero, focus-target, formation
  (line/wedge/loose), stance (aggressive/defensive).
- Win: destroy enemy hero + army rout threshold, or points majority at timeout.
- Action space: `hero_action (move/ability/attack)` every tick + `command (squad_id, verb,
  target)` rate-limited (e.g., 1 per 5 ticks) — the hierarchy is IN the interface.
- Duplicate eval: mirrored armies + mirrored map; pair-aggregate scoring (docs/04).

## Phase 2 (P4): Conquest campaign — the open-world loop
- Persistent async world map (M&B-style): fiefs, recruitment pools, economy; players
  (agents or humans) hold territory; battles are scheduled ladder matches whose stakes are
  campaign resources.
- Between battles: army composition, upgrades, terrain choice = strategic meta-decisions
  (slow, turn-based, async — no real-time server burden).
- Season = one campaign; season end = dataset release + paper-grade natural experiment.
- LLM-agent hook: the campaign layer (negotiation, alliances, logistics in natural
  language) is where language agents enter the platform.

## Deliberate constraints
- 2D top-down forever (readable, trainable, renderable in PixiJS).
- Battle layer must run vmapped ≥1k parallel battles/GPU before the campaign layer starts.
- Campaign state machine is content-light v1: 12 fiefs, 3 factions, weekly ticks.

## Open design questions (decide at P3 kickoff)
1. Hero death = battle loss, or respawn-with-penalty (MOBA-style)?
2. Squad micro fidelity: per-unit sim (Boom-style, expensive) vs squad-blob sim (cheap,
   recommended v1) — recommend squad-blob with unit-count as HP.
3. Fog of war in battle layer v1? (recommend: no fog v1; fog = v2 research knob)
4. Campaign persistence: fully on-chain-of-record Postgres vs event-sourced replay
   (recommend event-sourced — audit + research reuse).
